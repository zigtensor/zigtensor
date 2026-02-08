const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const types = @import("../types.zig");
const errors = @import("../errors.zig");

pub fn add(comptime T: type, allocator: std.mem.Allocator, a: *const Tensor(T), b: *const Tensor(T)) !Tensor(T) {
    for (a.shape, b.shape) |a_value, b_value| {
        if (a_value != b_value) {
            return errors.Error.ShapeMismatch;
        }
    }

    for (a.strides, b.strides) |a_value, b_value| {
        if (a_value != b_value) {
            return errors.Error.StrideMismatch;
        }
    }

    var out = try Tensor(T).initCpu(allocator, a.shape, a.strides, a.device_id, null, 0);
    errdefer out.deinit();

    for (out.data, a.data, b.data) |*dst, a_value_to_add, b_value_to_add| {
        dst.* = a_value_to_add + b_value_to_add;
    }

    return out;
}
