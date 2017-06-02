highlight_list = [];
var svgContainer = d3.select("#visualize_area").append("svg")
                                        .attr("width", "100%")
                                        .attr("height", "100%");
$( document ).ready(function(){
    make_app_name_list(app_name_list);  //app_name_list is in app_name.js
    make_app_embedding_visualization(highlight_list);
    $( "#search_app_input" ).focus();

});
$( "#search_app_input" ).keyup(function() {
    now_value = $(this).val();
    console.log(now_value);
    match_app_list = [];
    if(now_value.length == 0){
        make_app_name_list(app_name_list);
    }else{
        for(var i=0; i<app_name_list.length; i++){
            if(app_name_list[i].toLowerCase().indexOf(now_value.toLowerCase()) != -1){
                match_app_list.push(app_name_list[i]);
            }
        }
        make_app_name_list(match_app_list); 
    }
});

function make_app_name_list(show_list){
    // make show_list a vertical list in sidebar, which is also a button
    $("#app_list_area").empty();
    for(var i = 0; i < show_list.length; i++){
        var button_string = '<button type="button" onclick="add_to_highlight(this)" class="btn btn-block">'+show_list[i]+'</button>';
        var new_button = $(button_string);
        $("#app_list_area").append(new_button);
    }
    return;
}

function add_to_highlight(button){
    console.log(button.innerHTML);
    highlight_list.push(button.innerHTML);
    make_app_embedding_visualization(highlight_list);
}

function make_app_embedding_visualization(hl_list, init_flag){
    console.log(hl_list);
    // show all embedding in a transform type
    // hightlight chosen app in embedding
    // make svg or something


    // iterate through the map
    var max_x = app_emb["Facebook"][0], min_x = app_emb["Facebook"][0];
    var max_y = app_emb["Facebook"][1], min_y = app_emb["Facebook"][1];
    Object.keys(app_emb).forEach(function(key) {
        value = app_emb[key];
        x = value[0];
        y = value[1];
        if(x > max_x){
            max_x = x;
        }else if(x < min_x){
            min_x = x;
        }
        if(y > max_y){
            max_y = y;
        }else if(y < min_y){
            min_y = y;
        }
    });

    var value, circle, x, y, linear_x, linear_y;
    var height = $("#visualize_area").height()-8;
    var width = $("#visualize_area").width()-8;
    hl_dict = {};
    Object.keys(app_emb).forEach(function(key) {
        value = app_emb[key];
        x = value[0];
        y = value[1];
        linear_x  = (x-min_x)/(max_x-min_x)*width+4;
        linear_y  = (y-min_y)/(max_y-min_y)*height+4;
        if(hl_list.indexOf(key) !=-1){
            console.log("in hl_list");
            // 為了讓有上色的順序較後（才不會被白色擋住）
            hl_dict[key] = [linear_x, linear_y];
        
        }else{
            circle = svgContainer.append("circle").attr("cx", linear_x)
                                                    .attr("cy", linear_y)
                                                    .attr("r", 4)
                                                    .style("fill", "white");
        }
    });
    Object.keys(hl_dict).forEach(function(key) {
        x = hl_dict[key][0];
        y = hl_dict[key][1];
        console.log(key, x, y);
        circle = svgContainer.append("circle").attr("cx", x)
                                                .attr("cy", y)
                                                .attr("r", 4)
                                                .style("fill", "black");
        text = svgContainer.append("text").attr("x", x)
                                                .attr("y", y)
                                                .text(key)
                                                .attr("font-family", "sans-serif")
                                                .attr("font-size", "20px")
                                                .attr("fill", "red");
    });
        
    return;
}
