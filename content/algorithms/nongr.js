const C0_color = "#389ed0"
const C1_color = "#ef5946"
const C2_color = "#06b15c"
const C3_color = "#fab54e"
const C4_color = "#9d67a2"
const C5_color = "#545e56"

window.onload = function() {
  new Step0("#plot1", 600, 40, {top: 10, right: 30, bottom: 30, left: 30});
  new Step1("#plot2", 600, 40, {top: 30, right: 30, bottom: 30, left: 30});
};

var linspace = function(start, stop, nsteps){
  delta = (stop-start)/(nsteps-1)
  return d3.range(nsteps).map(function(i){return start+i*delta;});
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
      .domain([-2, 2])
      .range([0, this.ax_width])

    var xAxis = d3
                .axisBottom(this.xScale)
                  .tickValues(d3.range(-2, 3))
                  .tickFormat(x => (x == 0 ? 'n' : (x < 0) ? 'n' + x : 'n+' + x))
    this.svg
      .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + this.downY + ")")
        .call(xAxis)

    var xAxisFields = d3
                .axisBottom(this.xScale)
                  .tickValues(d3.range(-2, 3))
                  .tickFormat(x => '')
    this.svg
      .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + this.upY + ")")
        .call(xAxisFields)
  }
  addPoint(x, y, symbol, color, size, label=null, opacity=1.0, dy=-10) {
    var text = null
    if (label != null) {
      text = this.svg
        .append("text")
          .classed("label", true)
          .html(label)
          .style("text-anchor", "middle")
          .style("opacity", opacity)
          .attr("transform", "translate(" + this.xScale(x) + "," + (y + dy) + ")")
    }
    var symbol = this.svg
      .append("path")
        .attr("d", d3.symbol().size(size).type(symbol))
        .attr("transform", "translate(" + this.xScale(x) + "," + y + ")")
        .style("fill", color)
        .style("opacity", opacity)
    return [text, symbol]
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

    var t = this.addPoint(0, this.upY, d3.symbolStar, C1_color, 30, "B(n)")
    t[0].attr("transform", "translate(" + this.xScale(0) + "," + (this.upY-25) + ")");

    let _self = this;
    this.svg
      .append('path')
        .attr('d', function (d) {
          var source = {
            "x" : _self.xScale(-0.5),
            "y" : _self.upY
          };
          var target = {
            "x" : _self.xScale(0),
            "y" : _self.upY
          };
          var dx = target.x - source.x;
          var dy = target.y - source.y;
          var dr = Math.sqrt(dx * dx + dy * dy);
          return "M" +
              source["x"] + "," +
              source["y"] + "A" +
              dr + "," + dr + " 0 0,1 " +
              target["x"] + "," +
              target["y"]}
            )
      .attr('marker-end', 'url(#arrow)')
      .style("fill", "none")
      .attr("stroke", "black")

  }
}
