//! Direct comparison of the two logits kernels on the same buffers: the glue's sum in d order and the tensor units'.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const tokenizer = @import("tokenizer");
const sources = @import("inkling2_metal");
const Model = @import("model.zig").Model;
const backend = @import("backend.zig");
const st = @import("state.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const hidden = st.hidden;
const row_counts = [_]u32{ 1, 8, 16, st.max_rows };
const taps = [_]u32{ 5, 23, 41, 48 };
/// Not a multiple of 32, so the last tile of vocabulary rows is part full.
const small_vocab = 4122;
const capacity = 512;

const text =
    \\A ring buffer keeps the newest rows of a stream in a fixed block of memory. The writer puts each new row at
    \\the slot after the last one and wraps to the first slot when it reaches the end, so the oldest row is the one
    \\it overwrites. A reader that wants the last n rows walks back from the newest slot, and when the walk passes
    \\the start of the block it carries on from the end. Nothing is ever moved, which is why the structure suits a
    \\model's attention cache: a layer with a window of 1,024 positions needs the 1,023 rows before the current one
    \\and no others, and a ring of that size holds exactly those.
    \\
    \\fn slot(position: usize, ring: usize) usize {
    \\    return position % ring;
    \\}
    \\
    \\fn kept(position: usize, window: usize) usize {
    \\    return @min(position, window - 1);
    \\}
    \\
    \\The two functions above are the whole of the bookkeeping. To save a stream and restore it later into a ring of
    \\another size, copy the kept rows by position and not by slot: the row at position p goes to slot p modulo the
    \\new ring, wherever it sat in the old one. A copy by slot is only right when both rings are the same size, and
    \\it carries rows that no later position will read. The question to ask of any cache is therefore which rows a
    \\future step can still reach, and the answer for a sliding layer is the window less one.
;

fn bf(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    return round(@bitCast(v));
}

/// A finite FP32's BF16 bits, round to nearest even.
fn round(bits: u32) u16 {
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn finite(bits: u32) bool {
    return bits & 0x7f80_0000 != 0x7f80_0000;
}

/// The gap between neighbouring BF16 values at a finite FP32's magnitude.
fn step(bits: u32) f32 {
    const exponent: i32 = @intCast(@max((bits >> 23) & 0xff, 1));
    return std.math.ldexp(@as(f32, 1), exponent - 134);
}

fn spread(rng: std.Random, low: i32, high: i32) f32 {
    const mantissa = 1 + rng.float(f32);
    return std.math.ldexp(if (rng.boolean()) mantissa else -mantissa, rng.intRangeAtMost(i32, low, high));
}

const Fill = enum { served, random, wide, cancelling, exact, nonfinite };

/// Rows of h and, for a small vocabulary, its weights.
fn fill(kind: Fill, x: []u16, weights: ?[]u16, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    switch (kind) {
        .served => unreachable,
        .random, .nonfinite => {
            for (x, 0..) |*v, i| v.* = bf((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
            if (weights) |w| for (w) |*v| {
                v.* = bf(rng.float(f32) * 0.5 - 0.25);
            };
            if (kind == .nonfinite) {
                x[0] = bf(std.math.nan(f32));
                x[x.len / 2] = bf(std.math.inf(f32));
                x[x.len - 1] = bf(-std.math.inf(f32));
            }
        },
        // Terms over 24 binades, so a few large ones decide the sum and the order decides its rounding.
        .wide => {
            for (x) |*v| v.* = bf(spread(rng, -12, 12));
            if (weights) |w| for (w) |*v| {
                v.* = bf(spread(rng, -10, 2));
            };
        },
        // Columns in pairs: a value and its negation against a weight and the next BF16 up, so neighbours cancel.
        .cancelling => {
            var i: usize = 0;
            while (i < x.len) : (i += 2) {
                x[i] = bf(@floatFromInt(1 + rng.uintLessThan(u32, 1023)));
                x[i + 1] = x[i] ^ 0x8000;
            }
            if (weights) |w| {
                i = 0;
                while (i < w.len) : (i += 2) {
                    w[i] = bf(0.25 + rng.float(f32) * 0.25);
                    w[i + 1] = w[i] + 1;
                }
            }
        },
        // Whole numbers times a power of two: every product and partial sum is exact, so any order gives one sum.
        .exact => {
            for (x, 0..) |*v, i| v.* = bf(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= x.len / 2) @as(f32, -1) else 1));
            if (weights) |w| for (w, 0..) |*v, i| {
                v.* = bf(@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0x1p-20);
            };
        },
    }
}

/// What a variant of a kernel stores: the FP32 sum, its BF16 rounding, or the capped logit as served.
const Variant = enum { sum, rounded, served };
const glue_store = "ink_cap(float(bfloat(dot)), 30.0f)";
const units_store = "30.0f * precise::tanh(precise::divide(float(bfloat(dot)), 30.0f))";

fn edited(source: []const u8, from: []const u8, variant: Variant) ![]u8 {
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, from));
    return std.mem.replaceOwned(u8, gpa, source, from, switch (variant) {
        .sum => "dot",
        .rounded => "float(bfloat(dot))",
        .served => from,
    });
}

/// `far` counts pairs more than one BF16 step of the larger magnitude apart, and `far_gap` is the widest of those.
const Stat = struct { differ: usize = 0, far: usize = 0, gap: f32 = 0, far_gap: f32 = 0 };

fn compare(a: []const u32, b: []const u32) Stat {
    var out: Stat = .{};
    if (std.mem.eql(u32, a, b)) return out;
    for (a, b) |p, q| {
        if (p == q) continue;
        out.differ += 1;
        if (!finite(p) or !finite(q)) continue;
        const gap = @abs(@as(f32, @bitCast(p)) - @as(f32, @bitCast(q)));
        if (gap > step(@max(p & 0x7fff_ffff, q & 0x7fff_ffff))) {
            out.far += 1;
            out.far_gap = @max(out.far_gap, gap);
        }
        out.gap = @max(out.gap, gap);
    }
    return out;
}

fn best(row: []const u32) usize {
    var at: usize = 0;
    var top = -std.math.inf(f32);
    for (row, 0..) |bits, i| {
        const v: f32 = @bitCast(bits);
        if (v > top) {
            top = v;
            at = i;
        }
    }
    return at;
}

/// Sums whose host rounding is not the kernel's own rounded value: zero says the variants are one kernel.
fn misrounded(sum: []const u32, rounded: []const u32) usize {
    var n: usize = 0;
    for (sum, rounded) |s, r| {
        if (finite(s) and @as(u32, round(s)) << 16 != r) n += 1;
    }
    return n;
}

fn launch(queue: mtl.Queue, g: *const glue.Pipelines, x: *const experts.Pipelines, units: bool, h: glue.Ref, w: glue.Ref, out: mtl.Buffer, rows: u32, vocab: u32) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    if (units) {
        try experts.logitsUnits(x, e, .{ .buf = h.buf, .off = h.off }, .{ .buf = w.buf, .off = w.off }, .{ .buf = out, .off = 0 }, rows, vocab);
    } else {
        glue.logits(g, e, h, w, .{ .buf = out, .off = 0 }, rows, vocab);
    }
    e.end();
    cb.commit();
    cb.wait();
}

test "The tensor-unit logits kernel stores the glue kernel's bytes at 1, 8, 16 and the most rows" {
    const dir = @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse return error.SkipZigTest;
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    if (model.experts.units == null) return error.SkipZigTest;
    const device = model.device;
    const vocab: u32 = model.config.text.unpadded_vocab;

    // The kernels: the model's own as served, and each with its store edited to the sum and to the rounded sum.
    var glues: [3]glue.Pipelines = undefined;
    var units: [3]experts.Pipelines = undefined;
    for ([_]Variant{ .sum, .rounded }, 0..) |variant, i| {
        const g = try edited(sources.glue, glue_store, variant);
        defer gpa.free(g);
        glues[i] = try glue.Pipelines.loadSource(device, g);
        const u = try edited(sources.experts_units, units_store, variant);
        defer gpa.free(u);
        units[i] = model.experts;
        try units[i].loadUnits(device, u);
    }
    glues[2] = model.glue;
    units[2] = model.experts;
    defer for (0..2) |i| {
        glues[i].deinit();
        for (units[i].units.?) |pipe| pipe.deinit();
    };

    // The final-norm rows of a real prompt's 256 positions: tap 48 is the logits kernel's input.
    const real = try device.buffer(st.max_rows * hidden * 2, mtl.ResourceOptions.shared);
    defer real.deinit();
    {
        const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
        defer gpa.free(path);
        var tok = try tokenizer.loadTokenizer(io, gpa, path);
        defer tok.deinit();
        const ids = try tok.encode(gpa, text);
        defer gpa.free(ids);
        try std.testing.expect(ids.len >= st.max_rows);
        const target = try backend.Metal.init(gpa, model, .{ .capacity = capacity, .streams = 1, .budget = 2 * st.cacheBytes(capacity) });
        defer target.deinit();
        const b = target.backend();
        try b.vtable.prepare_features.?(b.ptr, &taps);
        var s = try lanes.Stream.init(gpa, .{ .id = "compare", .prompt = ids[0..st.max_rows], .max_new = 8, .drafts = false, .logprobs = 3 });
        defer s.deinit(gpa);
        try b.prefill(&s);
        defer b.release(&s);
        const f = try b.features(&s, &taps, 0, st.max_rows);
        try std.testing.expectEqual(@as(u32, st.max_rows), f.rows);
        const held: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
        const bytes = held.slice(u8, @intCast(f.offset + @as(u64, f.rows) * f.row_bytes));
        const to = real.slice(u8, st.max_rows * hidden * 2);
        for (0..st.max_rows) |row| @memcpy(to[row * hidden * 2 ..][0 .. hidden * 2], bytes[@intCast(f.offset + row * f.row_bytes + 3 * hidden * 2)..][0 .. hidden * 2]);
    }

    const made = try device.buffer(st.max_rows * hidden * 2, mtl.ResourceOptions.shared);
    defer made.deinit();
    const small = try device.buffer(small_vocab * hidden * 2, mtl.ResourceOptions.shared);
    defer small.deinit();
    var outs: [2][3]mtl.Buffer = undefined;
    for (&outs) |*pair| for (pair) |*out| {
        out.* = try device.buffer(@as(usize, st.max_rows) * vocab * 4, mtl.ResourceOptions.shared);
    };
    defer for (outs) |pair| for (pair) |out| out.deinit();

    const unembed: glue.Ref = .{ .buf = model.weights.unembed.buffer, .off = model.weights.unembed.offset };
    const cases = [_]struct { name: []const u8, fill: Fill, checkpoint: bool }{
        .{ .name = "checkpoint unembedding, a prompt's final-norm rows", .fill = .served, .checkpoint = true },
        .{ .name = "checkpoint unembedding, random rows", .fill = .random, .checkpoint = true },
        .{ .name = "checkpoint unembedding, rows over 24 binades", .fill = .wide, .checkpoint = true },
        .{ .name = "random weights and rows", .fill = .random, .checkpoint = false },
        .{ .name = "weights and rows over many binades", .fill = .wide, .checkpoint = false },
        .{ .name = "cancelling pairs of columns", .fill = .cancelling, .checkpoint = false },
        .{ .name = "exact products and sums", .fill = .exact, .checkpoint = false },
        .{ .name = "NaN and infinities in the rows", .fill = .nonfinite, .checkpoint = false },
    };
    var served_differ: usize = 0;
    var inconsistent: usize = 0;
    for (cases, 0..) |case, index| {
        if (case.fill != .served) fill(case.fill, made.slice(u16, st.max_rows * hidden), if (case.checkpoint) null else small.slice(u16, small_vocab * hidden), 100 + index);
        const w: glue.Ref = if (case.checkpoint) unembed else .{ .buf = small, .off = 0 };
        const n: u32 = if (case.checkpoint) vocab else small_vocab;
        std.debug.print("\n{s} ({d} vocabulary rows)\n", .{ case.name, n });
        for (row_counts) |rows| {
            // A prompt's rows are its last ones, so one row is the row that picks the first new token.
            const h: glue.Ref = if (case.fill == .served) .{ .buf = real, .off = (st.max_rows - rows) * hidden * 2 } else .{ .buf = made, .off = 0 };
            const count = @as(usize, rows) * n;
            for (0..3) |v| for (0..2) |kernel| {
                // Sentinels that differ, so an element neither kernel stored counts as a difference.
                @memset(outs[kernel][v].slice(u32, count), @as(u32, 0x7fc0_0001) + @as(u32, @intCast(kernel)));
                try launch(model.queue, &glues[v], &units[v], kernel == 1, h, w, outs[kernel][v], rows, n);
            };
            var stats: [3]Stat = undefined;
            for (&stats, 0..) |*stat, v| stat.* = compare(outs[0][v].slice(u32, count), outs[1][v].slice(u32, count));
            var tops: usize = 0;
            for (0..rows) |row| {
                if (best(outs[0][2].slice(u32, count)[row * n ..][0..n]) != best(outs[1][2].slice(u32, count)[row * n ..][0..n])) tops += 1;
            }
            for (0..2) |kernel| inconsistent += misrounded(outs[kernel][0].slice(u32, count), outs[kernel][1].slice(u32, count));
            // A rounded value that differs must differ as served, unless the cap has flattened both.
            served_differ += stats[2].differ;
            std.debug.print("  rows {d:>3}: of {d:>8} logits the FP32 sums differ in {d} (largest gap {d:.6}); the BF16 results in {d} (largest gap {d:.6}), {d} of them by more than one BF16 step (the widest of those {d:.6}); the served FP32 logits in {d} (largest gap {d:.6}); the top id differs in {d} of {d} rows\n", .{
                rows, count, stats[0].differ, stats[0].gap, stats[1].differ, stats[1].gap, stats[1].far, stats[1].far_gap, stats[2].differ, stats[2].gap, tops, rows,
            });
        }
    }
    std.debug.print("\nsums whose host rounding is not the kernel's rounded value, both kernels, every case: {d}\n", .{inconsistent});
    try std.testing.expectEqual(@as(usize, 0), inconsistent);
    try std.testing.expectEqual(@as(usize, 0), served_differ);
}
