const std = @import("std");
const zigtensor = @import("zigtensor");
const errors = @import("../../errors.zig");

const Tensor = zigtensor.Tensor;

pub fn allocAdd(first: Tensor, second: Tensor, allocator: *std.mem.Allocator) !Tensor {
    if (!std.mem.eql(usize, first.shape, second.shape)) {
        return errors.Error.ShapeMismatch;
    }
    if (!std.mem.eql(usize, first.strides, second.strides)) {
        return errors.Error.StrideMismatch;
    }

    var t = try Tensor(Tensor.type).initCpu(
        &allocator,
        first.type,
        first.shape,
        first.strides,
        first.device_id,
        first.slice()
    );

    for (second.data, 0..) |value_to_add, i| {
        t.data[i] += value_to_add;
    }

    return t;
}
