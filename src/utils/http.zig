const std = @import("std");

pub fn httpGet(gpa: std.mem.Allocator, url: []const u8) ![]u8 {
    var client: std.http.Client = .{ .allocator = gpa, .io = std.Io.Threaded.global_single_threaded.io() };
    defer client.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();

    const response = try client.fetch(.{ .method = .GET, .location = .{ .url = url }, .response_writer = &aw.writer });

    if (response.status != .ok) {
        return error.BadStatusCode;
    }

    var list = aw.toArrayList();
    return try list.toOwnedSlice(gpa);
}
