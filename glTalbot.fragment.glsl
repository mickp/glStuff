uniform float time;
uniform float wavelength;
uniform vec2 resolution;
uniform float propscale=1;
uniform vec2 aperture[128];

void main( void ) {

vec2 position = vec2(0., -1.) + 2.0 * gl_FragCoord.xy / resolution.xy;
position *= vec2(propscale, 1.);
float s = 0;
float r;
for (int i=0;i<aperture.length();i++){
    r = sqrt(pow(position.y-aperture[i].x, 2) + pow(position.x, 2));
    float x = r * 2 * 3.141 / wavelength;
    s += aperture[i].y * (cos(x-time) + sin(x-time) ) / (2.7183+r);
 }
s /= aperture.length() / (4*sqrt(abs(aperture[0].x)));
gl_FragColor = vec4(2*s, 0, -2*s, 1.);
}