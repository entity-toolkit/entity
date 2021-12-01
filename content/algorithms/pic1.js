window.onload = function() {
  var parentDiv = "#plot"
  var plot = new ShapePlot("#plot", 600, 80)
  MathJax.typeset()
};

var linspace = function(start, stop, nsteps){
  delta = (stop-start)/(nsteps-1)
  return d3.range(nsteps).map(function(i){return start+i*delta;});
}

class ShapePlot {
  constructor(parent, w, h) {
    this.parent = parent;
    this.width = w;
    this.height = h;

    this.margin = {top: 10, right: 30, bottom: 20, left: 30};
    this.width = this.width - this.margin.left - this.margin.right;
    this.height = this.height - this.margin.top - this.margin.bottom;

    this.svg = d3.select(parent)
      .append("div")
        .classed("svg-container", true)
      .append("svg")
        .attr("preserveAspectRatio", "xMinYMin meet")
        .attr("viewBox", "0 0 600 400")
        .classed("svg-content-responsive", true)
      .append("svg")
        .attr("width", this.width + this.margin.left + this.margin.right)
        .attr("height", this.height + this.margin.top + this.margin.bottom)
      .append("g")
        .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");

    // build scales
    this.xScale = d3.scaleLinear()
      .domain([-2, 2])
      .range([0, this.width])

    this.svg
      .append("p")
      .text("$ax^2 + bx$")

      // .text(
      //   function() {
      //     setTimeout(function(){MathJax.Hub.Queue(["Typeset",MathJax.Hub]);}, 10);
      //     return "$ax^2 + bx$";
      //   }
      // )

    var xAxis = d3
                .axisBottom(this.xScale)
                  .tickValues(d3.range(-2, 3))
                  .tickFormat(x => (x == 0 ? 'n' : (x < 0) ? 'n' + x : 'n+' + x))
    this.svg
      .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0," + this.height + ")")
        .call(xAxis)

    var xAxisFields = d3
                .axisBottom(this.xScale)
                  .tickValues(d3.range(-2, 3))
                  .tickFormat(x => '')
    this.svg
      .append("g")
        .attr('class', 'axes')
        .attr("transform", "translate(0,10)")
        .call(xAxisFields)

    var _self = this;
  }
}
