$(document).ready(function () {
    $("#hideshow").click(function() {
        // assumes element with id='button'
        $("#content").toggle();
    });

    $('#neg_option_clear').click(function() {
        $('.rel_option').prop('checked', false);
    });
});