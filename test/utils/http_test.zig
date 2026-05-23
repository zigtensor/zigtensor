const std = @import("std");
const zigtensor = @import("zigtensor");

const httpClientGet = zigtensor.utils.httpGet;

test "expect to fetch mnist" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();

    const url = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset";

    const data = try httpClientGet(gpa.allocator(), url);
    defer gpa.allocator().free(data);

    std.debug.print("Got {} bytes", .{data.len});
}
