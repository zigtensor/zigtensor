const std = @import("std");
const zigtensor = @import("zigtensor");

const Tensor = zigtensor.Tensor;
const DType = zigtensor.DType;
const Device = zigtensor.Device;

test "expect tensor shape to be {2,3}" {
    var gpa = std.heap.GeneralPurposeAllocator({}){};
    defer gpa.deinit();
    const allocator = gpa.allocator();

    const t = Tensor.init(
        allocator,
        &[_]usize{2,3},
        &[_]usize{3,1},
        zigtensor.DType.f32,
        zigtensor.Device.CPU,
        null,
    );

    defer t.deinit();

    try std.testing.expect(t.shape.len == 2);
    try std.testing.expect(t.data.len == 7);
}
