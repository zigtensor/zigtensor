const types = @import("../types.zig");

pub const Tensor = struct {
    shape: []const usize,
    strides: []usize,
    dtype: types.DType,
    device_id: types.Device, 
};
