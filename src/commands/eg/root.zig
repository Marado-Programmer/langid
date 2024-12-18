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

const Params = struct {
    example: []const u8 = undefined,
    input_samples: usize = undefined,
    layers: ?[]usize = null,
    learning_rate: f32 = 1e-3,
    learning_amount: usize = 1e2,
    randomize: bool = true,
};
const ParamsSpecificationError = error{ InvalidParamValue, ParamRepetition };

pub fn main(allocator: std.mem.Allocator, args: [][]const u8) !void {
    var params = Params{};
    defer if (params.layers) |layers| allocator.free(layers);

    params.example = args[0];
    params.input_samples = try std.fmt.parseInt(usize, args[1], 10);

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
    if (std.mem.eql(u8, params.example, "parabola")) {
        training_data = (try allocator.alloc([]f32, params.input_samples));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 5);
        }
        allocated = true;
        @import("./parabola.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, params.example, "linear")) {
        training_data = (try allocator.alloc([]f32, params.input_samples));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 4);
        }
        allocated = true;
        @import("./linear.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, params.example, "doubles")) {
        training_data = (try allocator.alloc([]f32, params.input_samples));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 2);
        }
        allocated = true;
        @import("./doubles.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, params.example, "abs")) {
        training_data = (try allocator.alloc([]f32, params.input_samples));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 2);
        }
        allocated = true;
        @import("./abs.zig").generate_data(&training_data, rand);
    } else if (std.mem.eql(u8, params.example, "sin")) {
        training_data = (try allocator.alloc([]f32, params.input_samples));
        for (training_data, 0..) |_, i| {
            training_data[i] = try allocator.alloc(f32, 2);
        }
        allocated = true;
        @import("./sin.zig").generate_data(&training_data, rand);
    } else {
        @panic("");
    }

    if (training_data.len < 1) {
        @panic("");
    }

    var skip = false;
    for (args[2..], 2..) |arg, i| {
        if (skip) {
            skip = false;
            continue;
        }

        if (std.mem.eql(u8, arg, "--layers")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const layers = args[i + 1];
            try specify_layers(allocator, &params, layers);
        } else if (std.mem.startsWith(u8, arg, "--layers=")) {
            const a = "--layers=";
            try specify_layers(allocator, &params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const learning_rate = args[i + 1];
            try specify_learning_rate(&params, learning_rate);
        } else if (std.mem.startsWith(u8, arg, "--learning-rate=")) {
            const a = "--learning-rate=";
            try specify_learning_rate(&params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--learning-amount")) {
            if (i + 1 >= args.len) {
                @panic("No method specified");
            }

            skip = true;
            const learning_amount = args[i + 1];
            try specify_learning_amount(&params, learning_amount);
        } else if (std.mem.startsWith(u8, arg, "--learning-amount=")) {
            const a = "--learning-amount=";
            try specify_learning_amount(&params, arg[a.len..]);
        } else if (std.mem.eql(u8, arg, "--randomize")) {
            params.randomize = true;
        } else if (std.mem.eql(u8, arg, "--no-randomize")) {
            params.randomize = false;
        }
    }

    var nn = try NeuralNetwork().init(allocator, training_data[0].len - 1, params.layers.?);
    defer nn.deinit(allocator);
    if (params.randomize) {
        nn.randomize(rand);
    } else {
        nn.reset();
    }
    nn.learning_rate = params.learning_rate;
    for (nn.layers[0 .. nn.layers.len - 1], 0..) |_, i| {
        nn.layers[i].set_activation_functions(.lReLU);
    }

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    const first = training_data[0];
    const first_in = try mat.Matrix(f32, 2).init(first, [_]usize{ 1, first.len - 1 });
    const first_out = first[first.len - 1];

    var i: usize = 0;
    var last_err: ?f32 = null;
    const tolerance = 1e-3;
    var stagnation_count: usize = 0;
    const max_stagnation_iterations = 5;

    var expected = try mat.Matrix(f32, 2).init_alloc(allocator, [_]usize{ 1, 1 });
    defer expected.deinit(allocator);

    while (i < params.learning_amount) : (i += 1) {
        for (training_data) |value| {
            const in = try mat.Matrix(f32, 2).init(value, [_]usize{ 1, value.len - 1 });

            expected.buf[0] = value[value.len - 1];
            _ = try nn.feed_forward(in, expected);
        }

        expected.buf[0] = first_out;
        const y = try nn.feed_forward(first_in, expected);

        const cur_err = nn.err();
        if (last_err != null and std.math.approxEqAbs(f32, cur_err, last_err.?, tolerance)) {
            stagnation_count += 1;
            if (stagnation_count >= max_stagnation_iterations) {
                break;
            }
        } else {
            stagnation_count = 0;
        }

        try stdout.print("{d}\t{d}\t{d}\n", .{ i, nn.err(), y.buf[0] });
        last_err = cur_err;

        nn.use_training();
    }

    try bw.flush(); // don't forget to flush!
}

fn specify_layers(allocator: std.mem.Allocator, params: *Params, layers: []const u8) !void {
    if (params.layers) |prev_layers| {
        allocator.free(prev_layers);
    }

    var itererator = std.mem.splitScalar(u8, layers, '-');
    var layers_collector = std.ArrayList(usize).init(allocator);
    defer layers_collector.deinit();
    var i: usize = 0;
    while (itererator.next()) |value| {
        try layers_collector.append(try std.fmt.parseInt(usize, value, 10));
        i += 1;
    }
    params.layers = try allocator.alloc(usize, i);
    std.mem.copyForwards(usize, params.layers.?, layers_collector.items);
}

fn specify_learning_rate(params: *Params, rate: []const u8) (std.fmt.ParseFloatError || ParamsSpecificationError)!void {
    params.learning_rate = try std.fmt.parseFloat(f32, rate);
}

fn specify_learning_amount(params: *Params, amount: []const u8) (std.fmt.ParseIntError || ParamsSpecificationError)!void {
    params.learning_amount = try std.fmt.parseUnsigned(usize, amount, 10);
}
