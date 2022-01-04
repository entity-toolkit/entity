window.addEventListener("load", function(event) {
  const color = getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('--body-font-color');
  const width = 800, height = 600;
  const panel_w = 0;
  const margins = [10, 10, 10, 10];

  const xlim = [0, 71];
  const ymax = 0.5 * (xlim[1] - xlim[0]) * (height - margins[1] - margins[3]) / (width - panel_w - margins[0] - margins[2]);
  const ylim = [-ymax, ymax];
  const dx_px = 280;

  const rmin = 1.0, rmax = 20.5;

  var XY2PX = function(x, y) {
    var px = panel_w + margins[0] + (width - panel_w - margins[0] - margins[2]) * (x - xlim[0]) / (xlim[1] - xlim[0]);
    var py = margins[1] + (height - margins[1] - margins[3]) * (y - ylim[0]) / (ylim[1] - ylim[0]);
    return [px, height - py];
  }
  var S2PX = function(s) {
    return s * (width - panel_w - margins[0] - margins[2]) / (xlim[1] - xlim[0]);
  }

  var sketch = function(p) {
    var cnv;
    var panel;
    var savePNGbutton;

    var slider_nx1, slider_nx2;
    var slider_r0, slider_h;

    var [oX, oY] = XY2PX(0, 0);

    p.setup = function() {
      panel = p.createDiv()
      panel.parent('plot_ax_01');
      cnv = p.createCanvas(width, height);
      cnv.parent('plot_ax_01');
      p.strokeWeight(0.5);
      p.rectMode(p.CORNER);
      p.stroke(color);
      p.textAlign(p.LEFT, p.CENTER);
      p.textFont('monospace');
      p.smooth();

      savePNGbutton = p.createButton('Save as png');
      savePNGbutton.parent('plot_ax_01');
      savePNGbutton.style('width', '120px');
      savePNGbutton.mousePressed(() => {
        p.pixelDensity(3.0);
        frame();
        p.save("ax_grid.png");
        p.pixelDensity();
      });

      let slider_nx1_div = p.createDiv('nx1:');
      slider_nx1_div.parent(panel);
      slider_nx1_div.style('display', 'inline');
      slider_nx1_div.style('margin-right', '15px');
      let slider_nx2_div = p.createDiv('nx2:');
      slider_nx2_div.parent(panel)
      slider_nx2_div.style('display', 'inline');
      slider_nx2_div.style('margin-right', '15px');
      let slider_r0_div = p.createDiv('r0:');
      slider_r0_div.parent(panel)
      slider_r0_div.style('display', 'inline');
      slider_r0_div.style('margin-right', '15px');
      let slider_h_div = p.createDiv('h:');
      slider_h_div.parent(panel)
      slider_h_div.style('display', 'inline');
      slider_h_div.style('margin-right', '15px');

      slider_nx1 = p.createSlider(2, 100, 32, 1);
      slider_nx1.style('width', '80px');
      slider_nx1.parent(slider_nx1_div);

      slider_nx2 = p.createSlider(2, 64, 16, 1);
      slider_nx2.style('width', '80px');
      slider_nx2.parent(slider_nx2_div);

      slider_r0 = p.createSlider(-1, 0.99, 0.0, 0.01);
      slider_r0.style('width', '80px');
      slider_r0.parent(slider_r0_div);

      slider_h = p.createSlider(-0.5, 0.99, 0.4, 0.01);
      slider_h.style('width', '80px');
      slider_h.parent(slider_h_div);
    };

    p.draw = function() {
      frame()
    };

    function frame() {
      var nx1 = slider_nx1.value();
      var nx2 = slider_nx2.value();
      var r0 = slider_r0.value();
      var h = slider_h.value();

      if (r0 < -0.1) {
        r0 = -0.1 * Math.exp(-8.0 * r0) / Math.exp(8.0 * 0.1);
      }

      var x1_min = Math.log(rmin - r0)
      var x1_max = Math.log(rmax - r0)

      p.clear();

      p.noFill();
      for (var i = 0; i <= nx1; i++) {
        var r = rmin + i * (rmax - rmin) / nx1;
        var r_px = S2PX(r);
        p.arc(oX, oY, 2 * r_px, 2 * r_px, -p.HALF_PI, p.HALF_PI);
      }
      for (var j = 0; j <= nx2; j++) {
        var theta = j * Math.PI / nx2;
        var [x1, y1] = XY2PX(rmin * Math.sin(theta), rmin * Math.cos(theta));
        var [x2, y2] = XY2PX(rmax * Math.sin(theta), rmax * Math.cos(theta));
        p.line(x1, y1, x2, y2);
      }

      var x1_min0 = Math.log(rmin)
      var x1_max0 = Math.log(rmax)
      for (var i = 0; i <= nx1; i++) {
        var x1 = x1_min0 + i * (x1_max0 - x1_min0) / nx1;
        var r_px = S2PX(Math.exp(x1));
        p.arc(oX + dx_px, oY, 2 * r_px, 2 * r_px, -p.HALF_PI, p.HALF_PI);
      }
      for (var j = 0; j <= nx2; j++) {
        var x2 = -1.0 + j * 2.0 / nx2;
        var theta = Math.acos(-x2);
        var [x1, y1] = XY2PX(rmin * Math.sin(theta), rmin * Math.cos(theta));
        var [x2, y2] = XY2PX(rmax * Math.sin(theta), rmax * Math.cos(theta));
        p.line(x1 + dx_px, y1, x2 + dx_px, y2);
      }

      for (var i = 0; i <= nx1; i++) {
        var x1 = x1_min + i * (x1_max - x1_min) / nx1;
        var r_px = S2PX(r0 + Math.exp(x1));
        p.arc(oX + 2 * dx_px, oY, 2 * r_px, 2 * r_px, -p.HALF_PI, p.HALF_PI);
      }
      for (var j = 0; j <= nx2; j++) {
        var x2 = j * p.PI / nx2;
        var theta = x2 + 2.0 * h * x2 * (p.PI - 2.0 * x2) * (p.PI - x2) / (p.PI * p.PI);
        var [x1, y1] = XY2PX(rmin * Math.sin(theta), rmin * Math.cos(theta));
        var [x2, y2] = XY2PX(rmax * Math.sin(theta), rmax * Math.cos(theta));
        p.line(x1 + 2 * dx_px, y1, x2 + 2 * dx_px, y2);
      }

      p.textSize(24);
      p.fill(color);
      p.text('nx1: ' + String(nx1) + ', nx2: ' + String(nx2), 20, 20);
      p.text('r0/r_min: ' + String(parseFloat(String(r0)).toFixed(2)), 2 * width / 3, 20);
      p.text('h: ' + String(h), 2 * width / 3, 45);
    }
  };

  const myp5 = new p5(sketch);

}, false);
