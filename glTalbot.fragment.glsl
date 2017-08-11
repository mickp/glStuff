uniform float time;
uniform float wavelength;
uniform vec2 resolution;
uniform vec2 aperture[32];


void main( void ) {

vec2 position = vec2(0., -1.) + 2.0 * gl_FragCoord.xy / resolution.xy;
float s = 0;
float r;
for (int i=0;i<aperture.length();i++){
    r = sqrt(pow(position.y-aperture[0].x, 2) + pow(position.x, 2));
    float x = r * 2 * 3.141 / 0.05;//wavelength;
    s += aperture[i].y * (cos(x-time) + sin(x-time)) / (exp(0)+r);
 }
 s /= 32;
gl_FragColor = vec4(s, s, s, 1.);
}