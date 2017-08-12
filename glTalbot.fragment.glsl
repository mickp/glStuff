uniform float time;
uniform float wavelength;
uniform vec2 resolution;
uniform vec2 aperture[128];

void main( void ) {

vec2 position = vec2(0., -1.) + 2.0 * gl_FragCoord.xy / resolution.xy;
float s = 0;
float r;
float b = 0;
for (int i=0;i<aperture.length();i++){
    r = sqrt(pow(position.y-aperture[i].x, 2) + pow(position.x, 2));
    float x = r * 2 * 3.141 / wavelength;
    s += aperture[i].y * (cos(x-time) + sin(x-time)) / (exp(0)+r);
    if (aperture[i].x >= position.y-5e-3 && aperture[i].x <= position.y+5e-3) {
        b += aperture[i].y * 0.2;
    }
 }
 s /= 0.5 * aperture.length()/2;
 if (position.x < 0.001) {
    gl_FragColor = vec4(s, s, s/2+b, 1.);
 } else {
    gl_FragColor = vec4(s, s, s, 1.);
 }
}