const std = @import("std");
const errors = @import("./errors.zig");

pub const types = @import("./types.zig");
pub const tensor = @import("./tensor/tensor.zig");
// pub const cops = @import("./ops/cpu/arithmetic.zig");

pub const Tensor = tensor.Tensor;
pub const DType = types.DType;
pub const Device = types.Device;

pub fn dummy() !void {}
