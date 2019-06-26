$(document).ready(function(){
    // Set the dimensions of the canvas / graph
    var margin = {top: 50, right: 20, bottom: 30, left: 100},
        width = 1200 - margin.left - margin.right,
        height = 800 - margin.top - margin.bottom;

    // Set the ranges
    var x = d3.scaleTime().range([0, width]);
    var y = d3.scaleLinear().range([height, 0]);

    // Define the axes
    var xAxis = d3.axisBottom()
        .scale(x);

    var yAxis = d3.axisLeft()
        .scale(y);


    // Define the line
    var valueline = d3.line()
        .x(function(d) { return new Date(d['Date Filed']); })
        .y(function(d) { return y(d['Offer Amount']); });

    // Adds the svg canvas
    var svg = d3.select("#ipo_scatter")
        .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)

          // Add the X Axis
        svg.append("svg:g")
            .attr("class", "axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(xAxis);

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height+margin.bottom)
            .style("text-anchor", "middle")
            .text("Time");

        // Add the Y Axis
        svg.append("svg:g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + (margin.left - 10) + ",0)")
            .call(yAxis);

        // Y axis label
          svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 5)
            .attr("x", - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Offer Amount");



    // Get the data
    d3.csv("data/processed/1997-04_2019_full_ipo_data.csv", function(data) {

        // Scale the range of the data
        x.domain(d3.extent(data, function(d) { return new Date(d['Date Filed']); }));
        y.domain([0, d3.max(data, function(d) { return d['Offer Amount']; })]);

        // Add the valueline path.
        svg.append("path")
            .attr("class", "line")
            .attr("d", valueline(data));

        // Add the scatterplot
        svg.selectAll("dot")
            .data(data)
          .enter().append("circle")
            .attr("r", function(d) { return y(d['Offer Amount']); })
            .attr("cx", function(d) { return new Date(d['Date Filed']); })
            .attr("cy", function(d) { return y(d['Offer Amount']); });

    });

});
