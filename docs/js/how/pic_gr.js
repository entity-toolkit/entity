const ifDocumentContains = (id, callback) => {
  const element = document.getElementById(id)
  if (element) {
    callback(element)
  }
}

ifDocumentContains("pic_scheme", (e) => {
  const C0_color = "#389ed0"
  const C1_color = "#ef5946"
  const C2_color = "#06b15c"
  const C3_color = "#fab54e"
  const C4_color = "#9d67a2"
  const C5_color = "#545e56"
  const C6_color = "#e22850"

  const C0_color_light = "#8bc6e4"
  const C1_color_light = "#f48a7c"
  const C2_color_light = "#3af899"
  const C3_color_light = "#fccd88"
  const C4_color_light = "#be9ac1"
  const C5_color_light = "#7d8c80"
  const C6_color_light = "#e95d7c"

  function makeid(length) {
    var result = '';
    var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var charactersLength = characters.length;
    for (var i = 0; i < length; i++) {
      result += characters.charAt(Math.floor(Math.random() *
        charactersLength));
    }
    return result;
  }
  
  window.addEventListener("load", function (event) {
    new GRStep0("#grplot0", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });

    new GRStep1_1("#grplot1_1", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep1_2("#grplot1_2", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep1_3("#grplot1_3", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });

    new GRStep2_1("#grplot2_1", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });

    new GRStep3("#grplot3", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });

    new GRStep4_1("#grplot4_1", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep4_2("#grplot4_2", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep4_3("#grplot4_3", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep4_4("#grplot4_4", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep4_5("#grplot4_5", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
    new GRStep4_6("#grplot4_6", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });

    new GRStep5("#grplot5", 600, 40, { top: 20, right: 30, bottom: 30, left: 30 });
  }, false);


  class GRSteps {
    constructor(parent, w, h, margins) {
      this.parent = parent;
      this.margin = margins
      this.height = 2 * this.margin.top + this.margin.bottom + 2 * h;
      this.width = w;
      this.ax_width = this.width - this.margin.left - this.margin.right

      this.upY = this.margin.top;
      this.midY = (this.margin.top + h);
      this.downY = (this.margin.top + 2 * h);

      this.svg = d3.select(parent)
        .append("svg")
        .classed("d3svg", true)
        .attr("preserveAspectRatio", "xMinYMin meet")
        .attr("viewBox", "0 0 " + this.width + " " + this.height)
        .classed("svg-content-responsive", true)
        .append("svg")
        .attr("width", this.width)
        .attr("height", this.height)
        .append("g")
        .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");

      // build scales
      this.xScale = d3.scaleLinear()
        .domain([-2, 2])
        .range([0, this.ax_width])

      this.addAxis(this.upY)
      this.addAxis(this.midY)
      this.addAxis(this.downY, (x => (x == 0 ? 'n' : (x < 0) ? 'n' + x : 'n+' + x)))
    }
    addAxis(y, format = (x => ''), imin = -1, imax = 2) {
      var xax = d3
        .axisBottom(this.xScale)
        .tickValues(d3.range(imin, imax))
        .tickFormat(format)
      this.svg
        .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + y + ")")
        .call(xax)
    }
    addText(x, y, text, opacity = 1.0, align = "middle") {
      text = this.svg
        .append("text")
        .classed("label", true)
        .html(text)
        .style("text-anchor", align)
        .style("opacity", opacity)
        .attr("transform", "translate(" + this.xScale(x) + "," + y + ")")
      return text
    }

    addPoint(x, y, symbol, color, size, label = null, opacity = 1.0, dy = -10) {
      var text = null
      if (label != null) {
        text = this.addText(x, y + dy, label, opacity)
      }
      var symbol = this.svg
        .append("path")
        .attr("d", d3.symbol().size(size).type(symbol))
        .attr("transform", "translate(" + this.xScale(x) + "," + y + ")")
        .style("fill", color)
        .style("opacity", opacity)
      return [text, symbol]
    }
    addArrow(x1, y1, x2, y2, color = "black", type = "arced") {
      var name = makeid(10)
      this.svg.append("svg:defs").selectAll("marker")
        .data(["arrowhead" + name])
        .enter().append("svg:marker")
        .attr("id", String)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 8)
        .attr("refY", 0)
        .attr("markerWidth", 7)
        .attr("markerHeight", 7)
        .attr("orient", "auto")
        .style("fill", color)
        .append("svg:path")
        .attr("d", "M0,-5L10,0L0,5");
      let _self = this;
      this.svg
        .append('path')
        .attr('d', function (d) {
          var source = {
            "x": _self.xScale(x1),
            "y": y1
          };
          var target = {
            "x": _self.xScale(x2),
            "y": y2
          };
          var dx = target.x - source.x;
          var dy = target.y - source.y;
          var dr = Math.sqrt(dx * dx + dy * dy);
          if (type == "arced") {
            return "M" + source["x"] + "," + source["y"] +
              "A" + dr + "," + dr + " 0 0,1 " + target["x"] + "," + target["y"]
          } else if (type == "arced_r") {
            return "M" + source["x"] + "," + source["y"] +
              "A" + dr + "," + dr + " 0 0,0 " + target["x"] + "," + target["y"]
          } else {
            return "M" + source["x"] + "," + source["y"] +
              "L" + target["x"] + "," + target["y"]
          }
        })
        .attr('marker-end', 'url(#arrowhead' + name + ')')
        .style("fill", "none")
        .attr("stroke", color)
    }
  }

  class GRStep0 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);

      this.addPoint(-1, this.upY, d3.symbolCircle, C0_color, 30, "D(n-1)", 0.3)
      this.addPoint(-1.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-3/2)", 0.3)

      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep1_1 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(-1, this.upY, d3.symbolCircle, C0_color, 30, "D(n-1)")
      this.addPoint(-1.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-3/2)")

      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)")
      this.addPoint(-1, this.midY, d3.symbolCircle, C1_color, 30, "B(n-1)")

      this.addArrow(-1.0, this.upY, -0.5, this.midY, C0_color_light, "arced_r")
      this.addArrow(0.0, this.upY, -0.5, this.midY, C0_color_light, "arced")

      this.addArrow(-1.5, this.upY, -1, this.midY, C1_color_light, "arced_r")
      this.addArrow(-0.5, this.upY, -1, this.midY, C1_color_light, "arced")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep1_2 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)")
      this.addPoint(-1, this.midY, d3.symbolCircle, C1_color, 30, "B(n-1)", 0.3)

      var t = this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "E(n-1/2)")
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.midY - 25) + ")")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep1_3 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)
      this.addPoint(-1, this.midY, d3.symbolCircle, C1_color, 30, "B(n-1)")

      var t = this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "E(n-1/2)")
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.midY - 25) + ")")

      this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "B(n)")

      this.addArrow(-1, this.midY, 0, this.midY, C1_color_light, "arced")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep2_1 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)

      this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "B(n)")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")

      this.addArrow(0, this.downY, 1, this.downY, C3_color_light, "arced_r")
      this.addArrow(-0.5, this.downY, 0.5, this.downY, C2_color_light, "arced_r")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep3 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)

      this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)")

      this.addArrow(0, this.downY, 0.5, this.upY, C6_color_light, "straight")
      this.addArrow(1, this.downY, 0.5, this.upY, C6_color_light, "straight")
      // this.addArrow(-0.5, this.downY, 0.5, this.downY, C2_color_light, "arced_r")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep4_1 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)

      this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "B(n)")
      var l = this.addPoint(0, this.midY, d3.symbolCircle, C0_color, 30, "E(n), H(n)")
      l[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.midY - 25) + ")")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)

      // this.addArrow(0, this.downY, 0.5, this.upY, C6_color_light, "straight")
      // this.addArrow(1, this.downY, 0.5, this.upY, C6_color_light, "straight")
      // this.addArrow(-0.5, this.downY, 0.5, this.downY, C2_color_light, "arced_r")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep4_2 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)

      this.addPoint(0, this.midY, d3.symbolCircle, C0_color, 30, "E(n)")
      var k = this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "H(n)", 0.3)
      k[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.midY - 25) + ")")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")

      this.addArrow(-0.5, this.upY, 0.5, this.upY, C1_color_light, "arced_r")
      // this.addArrow(1, this.downY, 0.5, this.upY, C6_color_light, "straight")
      // this.addArrow(-0.5, this.downY, 0.5, this.downY, C2_color_light, "arced_r")

      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep4_3 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)", 0.3)

      this.addPoint(0, this.midY, d3.symbolCircle, C6_color, 30, "J(n)")
      var k = this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "H(n)", 0.3)
      k[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.midY - 25) + ")")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)")
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")

      this.addArrow(-0.5, this.upY, 0.0, this.midY, C6_color_light, "arced_r")
      this.addArrow(0.5, this.upY, 0.0, this.midY, C6_color_light, "arced")
      var t = this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "J(n-1/2)")
      t[0].attr("transform", "translate(" + this.xScale(-0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep4_4 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(-0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n-1/2)")
      this.addPoint(0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n+1/2)")

      this.addPoint(0, this.midY, d3.symbolCircle, C6_color, 30, "J(n)")
      var k = this.addPoint(0, this.midY, d3.symbolCircle, C1_color, 30, "H(n)")
      k[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.midY - 25) + ")")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")

      this.addArrow(-0.5, this.midY, 0.5, this.midY, C0_color_light, "arced_r")
    }
  }

  class GRStep4_5 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      var k = this.addPoint(0.5, this.midY, d3.symbolCircle, C1_color, 30, "H(n+1/2)")
      k[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.midY - 25) + ")")

      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(0.5, this.midY, d3.symbolCircle, C0_color, 30, "D(n+1/2)")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")
    }
  }

  class GRStep4_6 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      var k = this.addPoint(0.5, this.midY, d3.symbolCircle, C1_color, 30, "H(n+1/2)")

      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)")
      this.addPoint(1, this.upY, d3.symbolCircle, C0_color, 30, "D(n+1)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")
      this.addArrow(0.0, this.upY, 1.0, this.upY, C0_color_light, "arced_r")
    }
  }

  class GRStep5 extends GRSteps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "D(n)", 0.3)
      this.addPoint(1, this.upY, d3.symbolCircle, C0_color, 30, "D(n+1)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)", 0.3)
      this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")

      var l = this.addPoint(0.5, this.upY, d3.symbolSquare, C6_color, 30, "J(n+1/2)", 0.3)
      l[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")
    }
  }
})