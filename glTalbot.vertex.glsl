#version 430
uniform float wavelength;
attribute vec3 position;

void main() {
    float a = wavelength;
    gl_Position = vec4(position, 1.);
}