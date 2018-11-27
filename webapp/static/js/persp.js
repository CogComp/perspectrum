$(document).ready(function(){
    $(".persp-arg-container").hide()
    $(".persp-title").click(function(){
        $(this).next(".persp-arg-container").toggle();
    });
});