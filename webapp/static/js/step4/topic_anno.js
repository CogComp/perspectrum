$(document).ready(function () {
    csrfSetup();
    $('#code').hide();
    $('#rel_option_clear').click(function() {
        $('.custom-checkbox').prop('checked', false);
    });
    $('#rel_option_submit').click(function (){
        disable_submit_button();
        submit();
    });
});

/**
 * Submit the annotations to backend
 */
function submit() {

    // Get all annotations
    let annos = [];
    let radios = $("input.option_yes");
    let checked = radios.filter(':checked');
    let num_claims = $('.persps-container').length;

    if (checked.length < num_claims) {
        alert("Please select at least one category for each sentence");
        return enable_submit_button();
    }

    checked.each(function() {
        let el_id = $(this).attr('id').split("-");
        let cid_id = el_id.pop();
        let topic = el_id.pop();

        annos.push([cid_id, topic])
    });

    let annos_json = JSON.stringify(annos);

    console.log(annos_json);
    $.post("/step4/api/submit_topic_annotation", {
        "annotations": annos_json
    }, success_callback).fail(enable_submit_button)
}

/**
 * Callback for submission
 */
function success_callback(res) {
    $('.rel_option_container').hide();
    $('#code').html("Comleted! Your completion code = " + res);
    $('#code').show();
}

/**
 * failure callback
 */
function disable_submit_button() {
    $('#rel_option_submit').prop('disabled', true);
}

function enable_submit_button() {
    $('#rel_option_submit').prop('disabled', false);
}