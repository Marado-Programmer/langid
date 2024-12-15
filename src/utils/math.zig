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
const exp = @import("std").math.exp;
pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + exp(-x));
}
pub fn sigmoidp(x: f32) f32 {
    return sigmoid(x) * (1 - sigmoid(x));
}

pub fn identity(x: f32) f32 {
    return x;
}
pub fn identityp(_: f32) f32 {
    return 1;
}
