const std = @import("std");
const zigtensor = @import("zigtensor");

const Tensor = zigtensor.Tensor;
const DType = zigtensor.DType;
const Device = zigtensor.Device;

test "expect tensor shape to be {2,3}" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ =  gpa.deinit();

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor.initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.DType.f32,
        zigtensor.Device.CPU,
        null,
    );

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 24);

    t.deinit();
}

test "expect tensor slice" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ =  gpa.deinit();

    var data = [_]u8{1,2,3,4,5,6};

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor.initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.DType.f32,
        zigtensor.Device.CPU,
        &data,
    );

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 24);
    try std.testing.expect(t.slice() == data);

    t.deinit();
}
