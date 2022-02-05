window.addEventListener("load", function (event) {
  var color_fg = getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('--body-font-color');
  var color_bg = getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('--body-background');
  if (color_bg == 'white') {
    color_bg = '#ffffff';
  }
  
  const width = 600, height = 600;
  var slider;

  const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  var Ci, N = 15;
  var points = [];

  function hexToRgb(hex) {
    return hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i).slice(1).map(function (x) {
      return parseInt(x, 16)
    })
  }

  function linspace(start, stop, num, endpoint = true) {
    const div = endpoint ? (num - 1) : num;
    const step = (stop - start) / div;
    return Array.from({ length: num }, (_, i) => start + step * i);
  }

  function sphericalToCartesian([r, theta, phi]) {
    return [
      r * Math.sin(theta) * Math.cos(phi),
      r * Math.sin(theta) * Math.sin(phi),
      r * Math.cos(theta)
    ];
  }


  function precomputePoints(R, N) {
    var phis = linspace(-Math.PI / 4, Math.PI / 4, N);
    for (let j = 0; j < phis.length; j++) {
      var th_min, th_max;
      th_min = Math.atan(-Math.cos(phis[j])) + Math.PI / 2;
      th_max = Math.atan(Math.cos(phis[j])) + Math.PI / 2;
      var thetas = linspace(th_min, th_max, N);
      for (let i = 0; i < thetas.length - 1; i++) {
        const [x1, y1, z1] = sphericalToCartesian([R, thetas[i], phis[j]]);
        const [x2, y2, z2] = sphericalToCartesian([R, thetas[i + 1], phis[j]]);
        points.push([[x1, y1, z1], [x2, y2, z2]]);
      }
    }
  }

  var sketch = function (p) {
    p.setup = () => {
      cnv = p.createCanvas(width, height, p.WEBGL);
      cnv.parent('plot_cubed_sphere');
      slider = p.createSlider(0, 255, 200, 1);
      precomputePoints(200, N);
    };

    function cubedSphereFace(N) {
      p.stroke(hexToRgb(COLORS[Ci]));

      var n = 0;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N - 1; j++) {
          p.push()
          if ((i == 0) || (i == N - 1)) {
            p.stroke(0, 0, 0);
            p.strokeWeight(2);
          } else {
            p.strokeWeight(1);
          }
          p.line(points[n][0][0], points[n][0][1], points[n][0][2], points[n][1][0], points[n][1][1], points[n][1][2]);
          p.pop();
          n++;
        }
      }
      p.push();
      n = 0;
      p.rotateX(Math.PI / 2);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N - 1; j++) {
          p.push();
          if ((i == 0) || (i == N - 1)) {
            p.stroke(0, 0, 0);
            p.strokeWeight(2);
          } else {
            p.strokeWeight(1);
          }
          p.line(points[n][0][0], points[n][0][1], points[n][0][2], points[n][1][0], points[n][1][1], points[n][1][2]);
          p.pop();
          n++;
        }
      }
      p.pop();
      Ci++;
    }

    p.draw = () => {
      Ci = 0;
      p.orbitControl();
      p.background(hexToRgb(color_bg));

      p.rotateX(Math.PI / 2);

      p.rotateZ(Math.PI / 2);
      cubedSphereFace(N);
      p.rotateZ(Math.PI / 2)
      cubedSphereFace(N);
      p.rotateZ(Math.PI / 2)
      cubedSphereFace(N);
      p.rotateZ(Math.PI / 2)
      cubedSphereFace(N);
      p.rotateY(Math.PI / 2)
      cubedSphereFace(N);
      p.rotateY(Math.PI)
      cubedSphereFace(N);

      var alpha = slider.value();

      p.lights();
      p.ambientMaterial(255, 255, 255, alpha);
      p.noStroke();
      p.sphere(199, 64, 64);
    }

  };

  const myp5 = new p5(sketch);

}, false);
