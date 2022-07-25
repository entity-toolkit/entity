window.addEventListener("load", function (event) {
  // const color = getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('--body-font-color');
  let color;

  const getColors = () => {
    let c = getComputedStyle(document.body).getPropertyValue('--md-default-fg-color');
    // console.log(c)
    return (c.split(',').length == 4) ? c.split(',').slice(0, 3).join() + ',1)' : c;
  }
  color = getColors();

  const width = 800, height = 600;
  const panel_w = 0;
  const margins = [10, 10, 10, 10];

  const xlim = [0, 71];
  const ymax = 0.5 * (xlim[1] - xlim[0]) * (height - margins[1] - margins[3]) / (width - panel_w - margins[0] - margins[2]);
  const ylim = [-ymax, ymax];
  const dx_px = 280;

  const rmin = 1.0, rmax = 20.5;

  const XY2PX = (x, y) => {
    var px = panel_w + margins[0] + (width - panel_w - margins[0] - margins[2]) * (x - xlim[0]) / (xlim[1] - xlim[0]);
    var py = margins[1] + (height - margins[1] - margins[3]) * (y - ylim[0]) / (ylim[1] - ylim[0]);
    return [px, height - py];
  }
  const S2PX = (s) => {
    return s * (width - panel_w - margins[0] - margins[2]) / (xlim[1] - xlim[0]);
  }

  const sketch = (p) => {
    var cnv;
    var savePNGbutton;

    var slider_nx1, slider_nx2;
    var slider_r0, slider_h;
    var input_nx1, input_nx2;
    var input_r0, input_h;

    var [oX, oY] = XY2PX(0, 0);
    function r0_UPScale(r0) {
      if (r0 < -0.1) {
        r0 = -0.1 * Math.exp(-8.0 * r0) / Math.exp(8.0 * 0.1);
      }
      return r0;
    }
    function r0_DWNScale(r0) {
      if (r0 < -0.1) {
        r0 = -Math.log(-r0 * Math.exp(8.0 * 0.1) / 0.1) / 8.0;
      }
      return r0;
    }

    p.setup = function () {
      const panel1 = p.createElement("tbody")
        .parent(p.createElement("table").parent('plot_ax_01').style('display', 'inline-block').style('margin-right', '25px'));
      const panel2 = p.createElement("tbody")
        .parent(p.createElement("table").parent('plot_ax_01').style('display', 'inline-block'));
      cnv = p.createCanvas(width, height);
      cnv.parent('plot_ax_01');
      p.strokeWeight(0.5);
      p.rectMode(p.CORNER);
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

      const slider_nx1_tr = p.createElement("tr")
        .parent(panel1)
        .style('display', 'block');
      p.createElement("th", "nx1:").parent(slider_nx1_tr)
      input_nx1 = p.createInput('32', 'number')
        .parent(p.createElement("td").parent(slider_nx1_tr))
        .input(() => {
          slider_nx1.value(input_nx1.value());
          frame();
        });
      slider_nx1 = p.createSlider(2, 100, 32, 1)
        .parent(p.createElement("td").parent(slider_nx1_tr))
        .input(() => {
          input_nx1.value(slider_nx1.value());
          frame();
        });

      const slider_nx2_tr = p.createElement("tr")
        .parent(panel1)
        .style('display', 'block');
      p.createElement("th", "nx2:").parent(slider_nx2_tr)
      input_nx2 = p.createInput('64', 'number')
        .parent(p.createElement("td").parent(slider_nx2_tr))
        .input(() => {
          slider_nx2.value(input_nx2.value());
          frame();
        });
      slider_nx2 = p.createSlider(2, 64, 16, 1)
        .parent(p.createElement("td").parent(slider_nx2_tr))
        .input(() => {
          input_nx2.value(slider_nx2.value());
          frame();
        });

      const slider_r0_tr = p.createElement("tr")
        .parent(panel2)
        .style('display', 'block');
      p.createElement("th", "r0:").parent(slider_r0_tr)
      input_r0 = p.createInput('0.0', 'number')
        .parent(p.createElement("td").parent(slider_r0_tr))
        .input(() => {
          slider_r0.value(r0_DWNScale(input_r0.value()));
          frame(undefined, undefined, Number(input_r0.value()), undefined);
        });
      slider_r0 = p.createSlider(-1, 0.99, 0.0, 0.01)
        .parent(p.createElement("td").parent(slider_r0_tr))
        .input(() => {
          input_r0.value(r0_UPScale(slider_r0.value()));
          frame();
        });

      const slider_h_tr = p.createElement("tr")
        .parent(panel2)
        .style('display', 'block');
      p.createElement("th", "h:").parent(slider_h_tr)
      input_h = p.createInput('0.4', 'number')
        .parent(p.createElement("td").parent(slider_h_tr))
        .input(() => {
          slider_h.value(input_h.value());
          frame();
        });
      slider_h = p.createSlider(-0.5, 0.99, 0.4, 0.01)
        .parent(p.createElement("td").parent(slider_h_tr))
        .input(() => {
          input_h.value(slider_h.value());
          frame();
        });
      frame();
    };

    const frame = (nx1 = slider_nx1.value(), nx2 = slider_nx2.value(), r0 = r0_UPScale(slider_r0.value()), h = slider_h.value()) => {
      color = getColors()
      p.stroke(color);

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

    p.draw = () => {
      if (getColors() != color) {
        frame();
      }
    }
  };

  const myp5 = new p5(sketch);

}, false);
