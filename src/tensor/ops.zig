const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const types = @import("../types.zig");
const errors = @import("../errors.zig");

pub fn add(comptime T: type, allocator: std.mem.Allocator, a: *const Tensor(T), b: *const Tensor(T)) !Tensor(T) {
    if (a.shape != b.shape) {
        return errors.Error.ShapeMismatch;
    }
    if (a.strides != b.strides) {
        return errors.Error.StrideMismatch;
    }
    return Tensor(f32).initCpu(allocator, a.shape, a.strides, a.device_id, a.add(b));
}
