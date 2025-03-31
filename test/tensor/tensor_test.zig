const std = @import("std");
const tensor = @import("../../src/tensor/tensor.zig");
const types = @import("../../src/types.zig");

test "expect tensor shape to be {2,3}" {
    const allocator = std.heap.GeneralPurposeAllocator(.{});

    const t = tensor.Tensor.init(
        allocator,
        &[_]usize{2,3},
        &[_]usize{3,1},
        types.DType.f32,
        types.Device.CPU,
        null,
    );

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 7);
}
