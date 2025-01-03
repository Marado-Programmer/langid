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

const Method = enum { bag_of_words };
const Params = struct { input: ?std.fs.File = null, method: Method = Method.bag_of_words, sort: bool = false };
const ParamsSpecificationError = error{ InvalidParamValue, ParamRepetition };

pub fn main(allocator: std.mem.Allocator, args: [][]const u8) !void {
    var params = Params{};
    defer if (params.input) |in| in.close();

    var skip = false;
    for (args, 0..) |arg, i| {
        if (skip) {
            skip = false;
            continue;
        }

        if (std.mem.eql(u8, arg, "--method")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const method = args[i + 1];
            try specify_method(&params, method);
        } else if (std.mem.startsWith(u8, arg, "--method=")) {
            const a = "--method=";
            try specify_method(&params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--input")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const path_arg = args[i + 1];
            try specify_input(&params, path_arg);
        } else if (std.mem.startsWith(u8, arg, "--input=")) {
            const a = "--input=";
            try specify_input(&params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--sort")) {
            params.sort = true;
        } else if (std.mem.eql(u8, arg, "--no-sort")) {
            params.sort = false;
        }
    }

    if (params.input == null) {
        params.input = std.io.getStdIn();
    }

    switch (params.method) {
        .bag_of_words => try @import("bow.zig").create_vocab(allocator, params.input.?, params.sort),
    }
}

fn specify_method(params: *Params, method: []const u8) ParamsSpecificationError!void {
    if (std.mem.eql(u8, method, "bag_of_words") or std.mem.eql(u8, method, "BoW")) {
        params.method = Method.bag_of_words;
    } else {
        return ParamsSpecificationError.InvalidParamValue;
    }
}

fn specify_input(params: *Params, path_arg: []const u8) (std.posix.RealPathError || std.fs.File.OpenError || ParamsSpecificationError)!void {
    var path_buffer: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fs.realpath(path_arg, &path_buffer);

    if (params.input) |in| in.close();

    params.input = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
}
