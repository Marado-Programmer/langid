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
const NeuralNetwork = @import("../../utils/neural.zig").Network;
const mat = @import("../../utils/matrix.zig");

pub fn main(allocator: std.mem.Allocator, args: [][]const u8) !void {
    if (args.len < 3) {
        @panic("");
    }

    const example = args[0];
    const data_size = try std.fmt.parseInt(usize, args[1], 10);

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    var training_data: [][]f32 = undefined;
    var allocated = false;
    defer if (allocated) {
        for (training_data) |value| {
            allocator.free(value);
        }
        allocator.free(training_data);
    };
    if (std.mem.eql(u8, example, "parabola")) {
        training_data = (try allocator.alloc([]f32, data_size));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 5);
        }
        allocated = true;
        @import("./parabola.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, example, "linear")) {
        training_data = (try allocator.alloc([]f32, data_size));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 4);
        }
        allocated = true;
        @import("./linear.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, example, "doubles")) {
        training_data = (try allocator.alloc([]f32, data_size));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 2);
        }
        allocated = true;
        @import("./doubles.zig").generate_data(&training_data, rand);
    } else {
        @panic("");
    }

    if (training_data.len < 1) {
        @panic("");
    }

    var layers_neurons = try allocator.alloc(usize, args.len - 2);
    defer allocator.free(layers_neurons);
    for (args[2..], 0..) |value, i| {
        layers_neurons[i] = try std.fmt.parseInt(usize, value, 10);
    }

    var nn = try NeuralNetwork().init(allocator, training_data[0].len - 1, layers_neurons);
    defer nn.deinit(allocator);
    nn.randomize(rand);
    nn.learning_rate = 1e-6;

    for (training_data, 0..) |value, a| {
        const in = try mat.Matrix(f32, 2).init(value, [_]usize{ 1, value.len - 1 });

        const out = try nn.feed_forward(in, value[value.len - 1]);

        if (a < 5) {
            //std.log.debug("ax^2 + bx + c = y <=> {d}({d})^2 + {d}({d}) + {d} = {d} ({d})", .{ value[0], value[3], value[1], value[3], value[2], out.buf[0], value[4] });
            std.log.debug("ax + b = y <=> {d}({d}) + {d} = {d} ({d})", .{ value[0], value[2], value[1], out.buf[0], value[3] });
            //std.log.debug("2x = y <=> 2({d}) = {d} ({d})", .{ value[0], out.buf[0], value[1] });
        }
    }

    var e = nn.err();
    nn.use_training();

    var i: usize = 0;
    while (i < 1e3) : (i += 1) {
        for (training_data) |value| {
            const in = try mat.Matrix(f32, 2).init(value, [_]usize{ 1, value.len - 1 });

            _ = try nn.feed_forward(in, value[value.len - 1]);
        }

        if (e == nn.err()) {
            break;
        }

        e = nn.err();
        nn.use_training();
    }

    std.log.debug("####################################################", .{});

    for (training_data, 0..) |value, a| {
        const in = try mat.Matrix(f32, 2).init(value, [_]usize{ 1, value.len - 1 });

        const out = try nn.feed_forward(in, value[value.len - 1]);

        if (a < 5) {
            //std.log.debug("ax^2 + bx + c = y <=> {d}({d})^2 + {d}({d}) + {d} = {d} ({d})", .{ value[0], value[3], value[1], value[3], value[2], out.buf[0], value[4] });
            std.log.debug("ax + b = y <=> {d}({d}) + {d} = {d} ({d})", .{ value[0], value[2], value[1], out.buf[0], value[3] });
            //std.log.debug("2x = y <=> 2({d}) = {d} ({d})", .{ value[0], out.buf[0], value[1] });
        }
    }
}
