if (document.getElementById("pic_scheme")) {
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

  window.addEventListener("load", () => {
    const width = document.getElementsByTagName("article")[0].offsetWidth
    const factor = width / 600
    const height = 40 * factor
    const margins = { top: 30 * factor, right: 30 * factor, bottom: 35 * factor, left: 30 * factor };
    new Step0("#plot0", width, height, margins);

    new Step1("#plot1", width, height, margins);

    new Step2_1("#plot2_1", width, height, margins);
    new Step2_2("#plot2_2", width, height, margins);

    new Step3_1("#plot3_1", width, height, margins);
    new Step3_2("#plot3_2", width, height, margins);

    new Step4("#plot4", width, height, margins);

    new Step5("#plot5", width, height, margins);

    new Step6("#plot6", width, height, margins);
  }, false);

  var linspace = (start, stop, nsteps) => {
    delta = (stop - start) / (nsteps - 1)
    return d3.range(nsteps).map((i) => start + i * delta);
  }

  class Steps {
    constructor(parent, w, h, margins) {
      this.parent = parent;
      this.margin = margins
      this.height = 2 * this.margin.top + this.margin.bottom + h;
      this.width = w;
      this.ax_width = this.width - this.margin.left - this.margin.right

      this.upY = this.margin.top;
      this.downY = (this.margin.top + h);

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
        .domain([-1.5, 1.5])
        .range([0, this.ax_width])

      var xAxis = d3
        .axisBottom(this.xScale)
        .tickValues(d3.range(-1, 3))
        .tickFormat(x => (x == 0 ? 'n' : (x < 0) ? 'n' + x : 'n+' + x))
      this.svg
        .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + this.downY + ")")
        .call(xAxis)

      var xAxisFields = d3
        .axisBottom(this.xScale)
        .tickValues(d3.range(-1, 3))
        .tickFormat(x => '')
      this.svg
        .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + this.upY + ")")
        .call(xAxisFields)
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

  class Step0 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")
    }
  }

  class Step1 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
      this.addPoint(-0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n-1/2)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)")
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addArrow(-0.5, this.upY, 0.0, this.upY, C1_color_light, "arced_r")
    }
  }

  // class Step2_1 extends Steps {
  //   constructor(parent, w, h, margins) {
  //     super(parent, w, h, margins);
  //     this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
  //     this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
  //     this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")
  //
  //     var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)")
  //     t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY-25) + ")")
  //     this.addArrow(0.0, this.upY, 0.0, this.downY, C4_color_light, "arced_r")
  //
  //     this.addText(0.1, 0.5 * (this.upY + this.downY), "E(x), B(x)", 1.0, "left")
  //   }
  // }

  class Step2_1 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)")
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addArrow(-0.5, this.downY, 0.5, this.downY, C2_color_light, "arced_r")
    }
  }

  // class Step2_2 extends Steps {
  //   constructor(parent, w, h, margins) {
  //     super(parent, w, h, margins);
  //     this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
  //     this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)")
  //     this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)
  //
  //     var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
  //     t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY-25) + ")")
  //
  //     this.addText(0.1, 0.5 * (this.upY + this.downY), "E(x), B(x)", 1.0, "left")
  //
  //     this.addArrow(-0.5, this.downY, -0.5, this.downY + 25, C2_color_light, "straight")
  //     this.addArrow(0.35, 0.5 * (this.upY + this.downY) + 5, 0.1, 0.5 * (this.upY + this.downY) + 45, C4_color_light, "straight")
  //
  //     this.addText(-0.5, this.downY + 35, "uc(n-1/2)", 1.0, "middle")
  //     this.addText(0.0, this.downY + 35, "Ec(x)", 1.0, "middle")
  //     this.addText(0.0, this.downY + 50, "Bc(x)", 1.0, "middle")
  //   }
  // }

  class Step2_2 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")
      this.addArrow(0, this.downY, 1, this.downY, C3_color_light, "arced_r")
    }
  }

  class Step2_3 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")

      this.addArrow(-0.5, this.downY + 20, 0.5, this.downY + 20, C2_color_light)

      this.addText(-0.5, this.downY + 35, "uc(n-1/2)", 1.0, "middle")
      this.addText(0.0, this.downY + 35, "Ec(x)", 1.0, "middle")
      this.addText(0.0, this.downY + 50, "Bc(x)", 1.0, "middle")

      this.addText(0.5, this.downY + 35, "uc(n+1/2)", 1.0, "middle")
    }
  }

  class Step2_4 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      // this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")

      this.addArrow(0.5, this.downY + 20, 0.5, this.downY, C2_color_light, "straight")

      // this.addText(-0.5, this.downY + 35, "uc(n-1/2)", 1.0, "middle")
      // this.addText(0.0, this.downY + 35, "Ec(x)", 1.0, "middle")
      // this.addText(0.0, this.downY + 50, "Bc(x)", 1.0, "middle")

      this.addText(0.5, this.downY + 35, "uc(n+1/2)", 1.0, "middle")
    }
  }

  class Step2_5 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      // this.addPoint(-0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n-1/2)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x(n)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addArrow(0, this.downY, 1, this.downY, C3_color_light, "arced_r")
    }
  }

  class Step3_1 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x~(n)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addArrow(1, this.downY, 0, this.downY, C3_color_light, "arced")
    }
  }

  class Step3_2 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
      this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x~(n)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      // this.addArrow(1, this.downY, 0, this.downY, C3_color_light, "arced")

      this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "J(n+1/2)")
      this.addArrow(0, this.downY, 0.5, this.upY, C6_color_light, "straight")
      this.addArrow(1, this.downY, 0.5, this.upY, C6_color_light, "straight")
    }
  }

  // class Step3_1 extends Steps {
  //   constructor(parent, w, h, margins) {
  //     super(parent, w, h, margins);
  //     this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
  //     this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
  //     this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x~(n)")
  //     this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")
  //
  //     var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
  //     t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY-25) + ")")
  //     this.addArrow(1, this.downY, 0, this.downY, C3_color_light, "arced")
  //   }
  // }
  //
  // class Step3_2 extends Steps {
  //   constructor(parent, w, h, margins) {
  //     super(parent, w, h, margins);
  //     this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
  //     this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
  //     this.addPoint(0, this.downY, d3.symbolSquare, C3_color, 30, "x~(n)")
  //     this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")
  //
  //     var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
  //     t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY-25) + ")")
  //     // this.addArrow(1, this.downY, 0, this.downY, C3_color_light, "arced")
  //
  //     this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "j~(n+1/2)")
  //     this.addArrow(0, this.downY, 0.5, this.upY, C6_color_light, "straight")
  //     this.addArrow(1, this.downY, 0.5, this.upY, C6_color_light, "straight")
  //   }
  // }

  class Step3_3 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "j(n+1/2)")
    }
  }

  class Step3_4 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)", 0.3)
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 0.3)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")
      this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "J(n+1/2)")
    }
  }


  class Step4 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      var t = this.addPoint(0, this.upY, d3.symbolCircle, C1_color, 30, "B(n)", 1.0)
      t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY - 25) + ")")

      var t = this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)", 1.0)
      t[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")
      this.addArrow(0.0, this.upY, 0.5, this.upY, C1_color_light, "arced_r")
      this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "J(n+1/2)", 0.3)
    }
  }


  class Step5 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(0, this.upY, d3.symbolCircle, C0_color, 30, "E(n)")
      this.addPoint(1, this.upY, d3.symbolCircle, C0_color, 30, "E(n+1)")
      this.addPoint(0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)", 0.3)
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)", 0.3)

      var t = this.addPoint(0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)", 1.0)
      t[0].attr("transform", "translate(" + this.xScale(0.5) + "," + (this.upY - 25) + ")")
      this.addPoint(0.5, this.upY, d3.symbolCircle, C6_color, 30, "J(n+1/2)")
      this.addArrow(0.0, this.upY, 1.0, this.upY, C0_color_light, "arced_r")
    }
  }

  class Step6 extends Steps {
    constructor(parent, w, h, margins) {
      super(parent, w, h, margins);
      this.addPoint(1, this.upY, d3.symbolCircle, C0_color, 30, "E(n+1)")
      this.addPoint(+0.5, this.upY, d3.symbolCircle, C1_color, 30, "B(n+1/2)")
      this.addPoint(+0.5, this.downY, d3.symbolSquare, C2_color, 30, "u(n+1/2)")
      this.addPoint(1, this.downY, d3.symbolSquare, C3_color, 30, "x(n+1)")
    }
  }
}