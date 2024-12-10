const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("leak");
    }
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    switch (args.len) {
        0 => unreachable,
        1 => std.debug.print(comptime help(), .{}),
        else => try run_command(allocator, args[1], args[2..]),
    }
}

fn help() []const u8 {
    return "HELP!";
}

fn run_command(allocator: std.mem.Allocator, command: []const u8, args: [][]const u8) !void {
    if (std.mem.eql(u8, command, "vocab")) {
        try @import("commands/vocab/root.zig").main(allocator, args);
        // } else if (std.mem.eql(u8, command, "model")) {
        //     try @import("commands/model/root.zig").main(allocator, args);
    } else if (std.mem.eql(u8, command, "help")) {
        std.debug.print(comptime help(), .{});
    } else {
        std.debug.print("no command was provided.\n", .{});
        std.debug.print(comptime help(), .{});
    }
}
