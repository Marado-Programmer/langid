// LANGuage IDentification - To identify languages.
// Copyright (C) 2024  João Augusto Costa Branco Marado Torres
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
// for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
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
    } else if (std.mem.eql(u8, command, "model")) {
        try @import("commands/model/root.zig").main(allocator, args);
    } else if (std.mem.eql(u8, command, "help")) {
        std.debug.print(comptime help(), .{});
    } else {
        std.debug.print("no command was provided.\n", .{});
        std.debug.print(comptime help(), .{});
    }
}
