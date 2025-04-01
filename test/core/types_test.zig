const std = @import("std");

const types = @import("zigtensor");

const DType = types.DType;
const Device = types.Device;

test "expect DType.f32 existence" {
    const tag_name = @tagName(DType.f32);
    try std.testing.expectEqualStrings("f32", tag_name);
}

test "expect DType.i32 existence" {
    const tag_name = @tagName(DType.i32);
    try std.testing.expectEqualStrings("i32", tag_name);
}

test "expect DType.bool existence" {
    const tag_name = @tagName(DType.bool);
    try std.testing.expectEqualStrings("bool", tag_name);
}

test "expect CPU device" {
    const tag_name = @tagName(Device.CPU);
    try std.testing.expectEqualStrings("CPU", tag_name);
}

test "expect GPU device" {
    const tag_name = @tagName(Device.GPU);
    try std.testing.expectEqualStrings("GPU", tag_name);
}
