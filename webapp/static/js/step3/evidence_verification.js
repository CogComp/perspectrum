$(document).ready(function () {
    csrfSetup();

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

    // Get all annotation
    let annos = [];
    let radios = $("input.equi_option");
    let checked = radios.filter(':checked');

    if (checked.length !== $(".equi-cand-container").length) {
        alert("Please finish all annotations!");
        return enable_submit_button();
    }

    checked.each(function() {
        let el_id = $(this).attr('id').split("_");
        let value = $(this).val();
        let persp_id = el_id.pop();
        let evi_id = el_id.pop();

        annos.push([evi_id, persp_id, value])
    });

    let annos_json = JSON.stringify(annos);

    let batch_id = $(location).attr('href').split('/').pop();
    $.post("/step3/api/submit_annotation", {
        "batch_id": batch_id,
        "annotations": annos_json
    }, success_callback).fail(enable_submit_button)
}

/**
 * Callback for submission
 */
function success_callback() {
    window.location.href = '/step3/task_list';
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