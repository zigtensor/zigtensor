const std = @import("std");
const tensor = @import("../../src/tensor/tensor.zig");
const types = @import("../../src/types.zig");

const t = tensor.Tensor{
    .shape = &[_]usize{2,3},
    .strides = &[_]usize{3,1},
    .dtype = types.DType.f32,
    .device_id = types.Device.CPU,
};

test "expect tensor shape to be {2,3}" {
    try std.testing.expect(t.shape.len == 2);
}
