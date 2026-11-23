//! Direct comparisons of the tensor-unit kernels with the kernels they replace, and of a prompt with each switched off.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const tokenizer = @import("tokenizer");
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const backend = @import("backend.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const experts = @import("experts.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const hidden = st.hidden;
const row_counts = [_]u32{ 1, 8, 16, st.max_rows };
const taps = [_]u32{ 5, 23, 41, 48 };
const capacity = 512;
const made = 64;
const topk = 8;

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
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The gap between neighbouring BF16 values at a BF16's magnitude.
fn step(bits: u16) f32 {
    const exponent: i32 = @intCast(@max((bits >> 7) & 0xff, 1));
    return std.math.ldexp(@as(f32, 1), exponent - 134);
}

fn spread(rng: std.Random, low: i32, high: i32) f32 {
    const mantissa = 1 + rng.float(f32);
    return std.math.ldexp(if (rng.boolean()) mantissa else -mantissa, rng.intRangeAtMost(i32, low, high));
}

/// `far` counts pairs more than one BF16 step of the larger magnitude apart, and `far_gap` is the widest of those.
const Stat = struct { n: usize = 0, differ: usize = 0, far: usize = 0, gap: f32 = 0, far_gap: f32 = 0 };

fn compare(a: []align(1) const u16, b: []align(1) const u16) Stat {
    var out: Stat = .{ .n = a.len };
    for (a, b) |p, q| {
        if (p == q) continue;
        out.differ += 1;
        if (p & 0x7f80 == 0x7f80 or q & 0x7f80 == 0x7f80) continue;
        const gap = @abs(value(p) - value(q));
        if (gap > step(@max(p & 0x7fff, q & 0x7fff))) {
            out.far += 1;
            out.far_gap = @max(out.far_gap, gap);
        }
        out.gap = @max(out.gap, gap);
    }
    return out;
}

fn show(what: []const u8, rows: u32, s: Stat) void {
    std.debug.print("  {s} rows {d:>3}: of {d:>8} BF16 values {d} differ (largest gap {d:.6}), {d} of them by more than one BF16 step (the widest of those {d:.6})\n", .{ what, rows, s.n, s.differ, s.gap, s.far, s.far_gap });
}

fn x(buf: mtl.Buffer) experts.Ref {
    return .{ .buf = buf, .off = 0 };
}

fn t(tensor: ck.Tensor) experts.Ref {
    return .{ .buf = tensor.buffer, .off = tensor.offset };
}

const Pass = struct {
    pool: mtl.objc.Pool,
    cb: mtl.CommandBuffer,
    e: mtl.ComputeEncoder,

    fn begin(queue: mtl.Queue) Pass {
        const pool = mtl.objc.Pool.push();
        const cb = queue.commandBuffer();
        return .{ .pool = pool, .cb = cb, .e = cb.compute(.serial) };
    }

    fn finish(p: Pass) void {
        p.e.end();
        p.cb.commit();
        p.cb.wait();
        p.pool.pop();
    }
};

/// A prompt's tap rows and its first `made` greedy tokens with the switches as they stand.
const Run = struct { rows: []u8, tokens: [made]u32 };

fn run(model: *Model, ids: []const u32) !Run {
    const target = try backend.Metal.init(gpa, model, .{ .capacity = capacity, .streams = 1, .budget = 2 * st.cacheBytes(capacity) });
    defer target.deinit();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps);
    var s = try lanes.Stream.init(gpa, .{ .id = "compare", .prompt = ids, .max_new = made, .drafts = false, .logprobs = 3 });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    const f = try b.features(&s, &taps, 0, @intCast(ids.len));
    try std.testing.expectEqual(@as(u32, @intCast(ids.len)), f.rows);
    const held: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
    const bytes = held.slice(u8, @intCast(f.offset + @as(u64, f.rows) * f.row_bytes));
    var out: Run = .{ .rows = try gpa.dupe(u8, bytes[@intCast(f.offset)..]), .tokens = undefined };
    errdefer gpa.free(out.rows);
    var next = try b.read(try b.first(&s, ids.len));
    out.tokens[0] = next;
    for (1..made) |i| {
        next = try b.read(try b.queue(&s, .{ .value = next }, ids.len + i));
        out.tokens[i] = next;
    }
    return out;
}

fn switches(e: bool, l: bool, d: bool, a: bool) void {
    fwd.cmp_experts_units = e;
    fwd.cmp_logits_units = l;
    fwd.cmp_prompt_dense = d;
    fwd.cmp_prompt_attn = a;
}

test "Each tensor-unit path stores the bytes of the path it replaces" {
    const dir = @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse return error.SkipZigTest;
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    if (model.experts.units == null) return error.SkipZigTest;
    const device = model.device;
    defer switches(true, true, true, true);

    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const all_ids = try tok.encode(gpa, text);
    defer gpa.free(all_ids);
    try std.testing.expect(all_ids.len >= st.max_rows);
    const ids = all_ids[0..st.max_rows];

    // 1. A whole prompt and 64 greedy tokens, every tensor-unit path of these pull requests off, then each on alone.
    switches(false, false, false, false);
    const base = try run(model, ids);
    defer gpa.free(base.rows);
    {
        const again = try run(model, ids);
        defer gpa.free(again.rows);
        try std.testing.expectEqualSlices(u8, base.rows, again.rows);
        try std.testing.expectEqualSlices(u32, &base.tokens, &again.tokens);
    }
    const configs = [_]struct { name: []const u8, e: bool, l: bool, d: bool, a: bool }{
        .{ .name = "the experts' products on the tensor units (#4523)", .e = true, .l = false, .d = false, .a = false },
        .{ .name = "the logits on the tensor units (#4618)", .e = false, .l = true, .d = false, .a = false },
        .{ .name = "a prompt's projections on the tensor units (#4631, first commit)", .e = false, .l = false, .d = true, .a = false },
        .{ .name = "a prompt's sliding attention on the prompt kernel (#4631, second commit)", .e = false, .l = false, .d = false, .a = true },
        .{ .name = "all four", .e = true, .l = true, .d = true, .a = true },
    };
    const row_bytes = taps.len * hidden * 2;
    var moved: usize = 0;
    std.debug.print("\nA 256-token prompt and {d} greedy tokens, against every path off\n", .{made});
    for (configs) |config| {
        switches(config.e, config.l, config.d, config.a);
        const got = try run(model, ids);
        defer gpa.free(got.rows);
        std.debug.print("{s}\n", .{config.name});
        for (taps, 0..) |tap, i| {
            var s: Stat = .{};
            for (0..ids.len) |row| {
                const at = row * row_bytes + i * hidden * 2;
                const one = compare(std.mem.bytesAsSlice(u16, base.rows[at..][0 .. hidden * 2]), std.mem.bytesAsSlice(u16, got.rows[at..][0 .. hidden * 2]));
                s.n += one.n;
                s.differ += one.differ;
                s.far += one.far;
                s.gap = @max(s.gap, one.gap);
                s.far_gap = @max(s.far_gap, one.far_gap);
            }
            std.debug.print("  after visit {d:>2}: of {d} BF16 values {d} differ (largest gap {d:.6}), {d} of them by more than one BF16 step\n", .{ tap, s.n, s.differ, s.gap, s.far });
            moved += s.differ;
        }
        var same: usize = 0;
        while (same < made and base.tokens[same] == got.tokens[same]) same += 1;
        std.debug.print("  greedy tokens: {d} of {d} the same before the first that differs\n", .{ same, made });
    }
    switches(true, true, true, true);

    // 2. The kernels themselves on the same buffers: real final-norm rows (tap 48 of the run above) and random rows.
    const real = try device.buffer(st.max_rows * hidden * 2, mtl.ResourceOptions.shared);
    defer real.deinit();
    for (0..st.max_rows) |row| @memcpy(real.slice(u8, st.max_rows * hidden * 2)[row * hidden * 2 ..][0 .. hidden * 2], base.rows[row * row_bytes + 3 * hidden * 2 ..][0 .. hidden * 2]);
    const random = try device.buffer(st.max_rows * hidden * 2, mtl.ResourceOptions.shared);
    defer random.deinit();
    const wide = try device.buffer(st.max_rows * hidden * 2, mtl.ResourceOptions.shared);
    defer wide.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (random.slice(u16, st.max_rows * hidden), 0..) |*v, i| v.* = bf((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
    for (wide.slice(u16, st.max_rows * hidden)) |*v| v.* = bf(spread(rng, -12, 12));
    // Each row's eight experts, ascending, as the router stores them.
    const routes = try device.buffer(st.max_rows * topk * 4, mtl.ResourceOptions.shared);
    defer routes.deinit();
    for (0..st.max_rows) |row| {
        var picked: [128]bool = @splat(false);
        var n: usize = 0;
        while (n < topk) {
            const id = rng.uintLessThan(u32, 128);
            if (picked[id]) continue;
            picked[id] = true;
            n += 1;
        }
        var at: usize = 0;
        for (picked, 0..) |on, id| if (on) {
            routes.slice(u32, st.max_rows * topk)[row * topk + at] = @intCast(id);
            at += 1;
        };
    }
    const gu_bytes = st.max_rows * topk * st.intermediate * 2;
    const down_bytes = st.max_rows * topk * hidden * 2;
    var gu: [2]mtl.Buffer = undefined;
    var down: [2]mtl.Buffer = undefined;
    for (&gu) |*b| b.* = try device.buffer(gu_bytes, mtl.ResourceOptions.shared);
    defer for (gu) |b| b.deinit();
    for (&down) |*b| b.* = try device.buffer(down_bytes, mtl.ResourceOptions.shared);
    defer for (down) |b| b.deinit();

    var kernel_differ: usize = 0;
    const inputs = [_]struct { name: []const u8, buf: mtl.Buffer }{ .{ .name = "a prompt's final-norm rows", .buf = real }, .{ .name = "random rows", .buf = random }, .{ .name = "rows over 24 binades", .buf = wide } };
    for ([_]bool{ true, false }) |nv| {
        var layer: usize = 0;
        while (layer < 24 and (model.weight_plan.at(@intCast(layer)).precision == .nvfp4) != nv) layer += 1;
        try std.testing.expect(layer < 24);
        const w = &model.weights.layers[layer];
        const format: experts.Format = if (nv) .nvfp4 else .mxfp8;
        for (inputs) |input| {
            std.debug.print("\nThe experts' products, layer {d} ({s}), {s}\n", .{ layer, if (nv) "NVFP4" else "MXFP8", input.name });
            for (row_counts) |rows| {
                const gu_call = experts.Call{ .rows = rows, .k = hidden, .n = st.intermediate, .format = format };
                const down_call = experts.Call{ .rows = rows, .k = st.intermediate, .n = hidden, .format = format };
                const gn = @as(usize, rows) * topk * st.intermediate;
                const dn = @as(usize, rows) * topk * hidden;
                @memset(gu[0].slice(u16, gn), 0x7fc1);
                @memset(gu[1].slice(u16, gn), 0x7fc2);
                @memset(down[0].slice(u16, dn), 0x7fc1);
                @memset(down[1].slice(u16, dn), 0x7fc2);
                // The kernels a round takes without tensor units: grouped from eight rows, a pair a threadgroup below.
                {
                    const p = Pass.begin(model.queue);
                    if (rows >= experts.grouped_rows) {
                        try experts.group(&model.experts, p.e, x(routes), gu_call);
                        try experts.guGrouped(&model.experts, p.e, x(input.buf), t(w.gu.weight), t(w.gu.scale), if (nv) .{ .amax = t(w.gu.amax.?), .denominator = t(w.gu.denominator.?) } else null, x(gu[0]), gu_call);
                    } else {
                        try experts.gu(&model.experts, p.e, x(input.buf), t(w.gu.weight), t(w.gu.scale), if (nv) .{ .amax = t(w.gu.amax.?), .denominator = t(w.gu.denominator.?) } else null, x(routes), x(gu[0]), gu_call);
                    }
                    p.finish();
                }
                {
                    const p = Pass.begin(model.queue);
                    try experts.group(&model.experts, p.e, x(routes), gu_call);
                    try experts.units(&model.experts, p.e, false, x(input.buf), t(w.gu.weight), t(w.gu.scale), if (nv) .{ .amax = t(w.gu.amax.?), .denominator = t(w.gu.denominator.?) } else null, x(gu[1]), gu_call);
                    p.finish();
                }
                // Both down kernels read the scalar path's activations.
                {
                    const p = Pass.begin(model.queue);
                    if (rows >= experts.grouped_rows) {
                        try experts.group(&model.experts, p.e, x(routes), down_call);
                        try experts.downGrouped(&model.experts, p.e, x(gu[0]), t(w.down.weight), t(w.down.scale), if (nv) .{ .amax = t(w.down.amax.?), .denominator = t(w.down.denominator.?) } else null, x(down[0]), down_call);
                    } else if (nv) {
                        try experts.nvfp4Down(&model.experts, p.e, x(gu[0]), t(w.down.weight), t(w.down.scale), t(w.down.amax.?), t(w.down.denominator.?), x(routes), x(down[0]), down_call);
                    } else {
                        try experts.mxfp8Down(&model.experts, p.e, x(gu[0]), t(w.down.weight), t(w.down.scale), x(routes), x(down[0]), down_call);
                    }
                    p.finish();
                }
                {
                    const p = Pass.begin(model.queue);
                    try experts.group(&model.experts, p.e, x(routes), down_call);
                    try experts.units(&model.experts, p.e, true, x(gu[0]), t(w.down.weight), t(w.down.scale), if (nv) .{ .amax = t(w.down.amax.?), .denominator = t(w.down.denominator.?) } else null, x(down[1]), down_call);
                    p.finish();
                }
                const g = compare(gu[0].slice(u16, gn), gu[1].slice(u16, gn));
                const d = compare(down[0].slice(u16, dn), down[1].slice(u16, dn));
                show("gate-up activations,", rows, g);
                show("down products,      ", rows, d);
                kernel_differ += g.differ + d.differ;
            }
        }
    }

    // A prompt's projections: the six of 2,304 columns on a visit's rows, and `wo` (3,072 columns) on random rows.
    const wide_in = try device.buffer(st.max_rows * 3072 * 2, mtl.ResourceOptions.shared);
    defer wide_in.deinit();
    for (wide_in.slice(u16, st.max_rows * 3072), 0..) |*v, i| v.* = bf((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
    var outs: [2]mtl.Buffer = undefined;
    for (&outs) |*b| b.* = try device.buffer(st.max_rows * 3072 * 2, mtl.ResourceOptions.shared);
    defer for (outs) |b| b.deinit();
    const attn = &model.weights.layers[0].attn;
    const projections = [_]struct { name: []const u8, w: ck.Tensor }{
        .{ .name = "wq", .w = attn.wq },     .{ .name = "wk", .w = attn.wk },     .{ .name = "wv", .w = attn.wv }, .{ .name = "wr", .w = attn.wr },
        .{ .name = "sink", .w = attn.sink }, .{ .name = "gate", .w = attn.gate }, .{ .name = "wo", .w = attn.wo },
    };
    for (inputs[0..2]) |input| {
        std.debug.print("\nA prompt's projections, layer 0, {s} (wo always on random rows)\n", .{input.name});
        for (projections) |projection| {
            const k = projection.w.shape[projection.w.rank - 1];
            const n = projection.w.shape[projection.w.rank - 2];
            const in = if (k == hidden) input.buf else wide_in;
            for (row_counts) |rows| {
                const count = @as(usize, rows) * n;
                @memset(outs[0].slice(u16, count), 0x7fc1);
                @memset(outs[1].slice(u16, count), 0x7fc2);
                {
                    const p = Pass.begin(model.queue);
                    fwd.dense(model, p.e, projection.w, .{ .buf = in, .off = 0 }, .{ .buf = outs[0], .off = 0 }, rows);
                    p.finish();
                }
                {
                    const p = Pass.begin(model.queue);
                    try experts.denseUnits(&model.experts, p.e, x(in), t(projection.w), x(outs[1]), rows, @intCast(n), @intCast(k));
                    p.finish();
                }
                const s = compare(outs[0].slice(u16, count), outs[1].slice(u16, count));
                show(projection.name, rows, s);
                kernel_differ += s.differ;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), kernel_differ + moved);
}
