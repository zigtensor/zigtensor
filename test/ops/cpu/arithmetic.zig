const std = @import("std");
const zigtensor = @import("zigtensor");

const Tensor = zigtensor.Tensor;
const allocAdd = zigtensor.cops.allocAdd;

test "allocAdd new Tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var data = [_]f32{1,2,3,4,5,6};

    var allocator = gpa.allocator();
    var stride = [_]usize{3,1};
    var t = try Tensor(f32).initCpu(
        &allocator,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        &data,
        null,
    );
    defer t.deinit();

    var gpa2 = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa2.deinit();

    var data2 = [_]f32{1,2,3,4,5,6};

    var allocator2 = gpa2.allocator();
    var t2 = try Tensor(f32).initCpu(
        &allocator2,
        &[_]usize{2,3},
        &stride,
        zigtensor.Device.CPU,
        &data2,
        null,
    );
    defer t2.deinit();

    var gpa3 = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa3.deinit();

    var t3 = allocAdd(t, t2, gpa3);

    const expected_data = [_]f32{2,4,6,8,10,12};
    const actual_data: []f32 = try t3.slice();

    try std.testing.expectEqualSlices(f32, expected_data[0..], actual_data);
}

