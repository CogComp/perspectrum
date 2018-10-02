$(document).ready(function () {
    $("#hideshow").click(function() {
        // assumes element with id='button'
        $("#content").toggle();
    });

    $('#rel_option_clear').click(function() {
        $('.rel_option').prop('checked', false);
    });
    $('#rel_option_submit').click(submit);
});

/**
 * Submit the annotations to backend
 */
function submit() {

    // Get all annotations
    let el_persps = $('.rel_radio_container');
    let annos = [];
    el_persps.each(function() {
        $(this).find('input:checked').each(function() {
            let el_id = $(this).attr('id').split("_");
            let persp_id = el_id.pop();
            let rel = el_id.pop();
            annos.push([persp_id, rel]);
        });
    });

    if (el_persps.length !== annos.length) {
        alert("Please annotate all questions!");
        return;
    }

    console.log('Success!')
}