highlight_list = [];
$( document ).ready(function(){
    make_app_name_list(app_name_list);  //app_name_list is in app_name.js
    make_app_embedding_visualization(highlight_list);
    $( "#search_app_input" ).focus();

});
$( "#search_app_input" ).keyup(function() {
    now_value = $(this).val();
    //console.log(now_value);
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
        if(highlight_list.indexOf(show_list[i]) != -1){
            button_string = '<button type="button" onclick="add_to_highlight(this)" class="btn btn-block btn-select">'+show_list[i]+'</button>';
            
        }
        var new_button = $(button_string);
        $("#app_list_area").append(new_button);
    }
    return;
}

function add_to_highlight(button){
    //console.log(button.innerHTML);
    var button_text = button.innerHTML;
    button_text = $("<div/>").html(button_text).text();
    if(highlight_list.indexOf(button_text) == -1){
        highlight_list.push(button_text);
        $(button).addClass("btn-select");
        make_app_embedding_visualization(highlight_list); 
    }else{
        var index = highlight_list.indexOf(button_text);
        if (index > -1) {
            highlight_list.splice(index, 1);
        }
        $(button).removeClass("btn-select");
        make_app_embedding_visualization(highlight_list);
    }
}

circles = [];

function make_app_embedding_visualization(hl_list, init_flag){
    
    $("#visualize_area").empty();
    var svgContainer = d3.select("#visualize_area").append("svg")
                                            .attr("width", "100%")
                                            .attr("height", "100%");
    //console.log(hl_list);
    // show all embedding in a transform type
    // highlight chosen app in embedding
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
            //console.log("in hl_list");
            // 為了讓有上色的順序較後（才不會被白色擋住）
            hl_dict[key] = [linear_x, linear_y];
        
        }else{
            var circle = svgContainer.append("circle").attr("cx", linear_x)
                                                    .attr("cy", linear_y)
                                                    .attr("r", 4)
                                                    .attr("key", key)
                                                    .style("fill", "white");
            circle.on("click", function(){
                circle_text = d3.select(this).attr("key");
                if(highlight_list.indexOf(circle_text) == -1){
                    highlight_list.push(circle_text);
                    $("#app_list_area > button").each(function(){
                        var button_text = this.innerHTML;
                        button_text = $("<div/>").html(button_text).text();
                        if(button_text === circle_text){
                            $(this).addClass("btn-select");
                        }
                    });
                    make_app_embedding_visualization(highlight_list); 
                }else{
                    var index = highlight_list.indexOf(circle_text);
                    if (index > -1) {
                        highlight_list.splice(index, 1);
                    }
                    $("#app_list_area > button").each(function(){
                        var button_text = this.innerHTML;
                        button_text = $("<div/>").html(button_text).text();
                        if(button_text === circle_text){
                            $(this).removeClass("btn-select");
                        }
                    });
                    make_app_embedding_visualization(highlight_list);
                }
                
            });
            circles.push(circle);
        }
    });
    Object.keys(hl_dict).forEach(function(key) {
        x = hl_dict[key][0];
        y = hl_dict[key][1];
        //console.log(key, x, y);
        text = svgContainer.insert("text", ":first-child").attr("x", x)
                                                .attr("y", y)
                                                .text(key)
                                                .attr("font-family", "sans-serif")
                                                .attr("font-size", "15px")
                                                .attr("fill", "red");
        var circle = svgContainer.append("circle").attr("cx", x)
                                                .attr("cy", y)
                                                .attr("r", 4)
                                                .attr("key", key)
                                                .style("fill", "black");
        circle.on("click", function(){
            circle_text = d3.select(this).attr("key");
            if(highlight_list.indexOf(circle_text) == -1){
                highlight_list.push(circle_text);
                $("#app_list_area > button").each(function(){
                    var button_text = this.innerHTML;
                    button_text = $("<div/>").html(button_text).text();
                    if(button_text === circle_text){
                        $(this).addClass("btn-select");
                    }
                });
                make_app_embedding_visualization(highlight_list); 
            }else{
                var index = highlight_list.indexOf(circle_text);
                if (index > -1) {
                    highlight_list.splice(index, 1);
                }
                $("#app_list_area > button").each(function(){
                    var button_text = this.innerHTML;
                    button_text = $("<div/>").html(button_text).text();
                    if(button_text === circle_text){
                        $(this).removeClass("btn-select");
                    }
                });
                make_app_embedding_visualization(highlight_list);
            }
            
        });
    });
        
    return;
}
