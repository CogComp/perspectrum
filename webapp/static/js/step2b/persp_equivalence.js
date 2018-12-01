$(document).ready(function () {
    csrfSetup();
    let submitted = false;
    let alerted = false;
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
    let annos = {};
    let radios = $("input.equi_option");
    let checked = radios.filter(':checked');

    if (checked.length !== $(".equi-cand-container").length) {
        alert("Please finish all annotations!");
        return enable_submit_button();
    }

    checked.each(function() {
        let el_id = $(this).attr('id').split("_");
        let cand_id = el_id.pop();
        let persp_id = el_id.pop();

        if (el_id.pop() === 'same') {
            if (persp_id in annos) {
                annos[persp_id].push(cand_id);
            }
            else {
                annos[persp_id] = [cand_id]
            }
        }

    });

    let annos_json = JSON.stringify(annos);

    let claim_id = $(location).attr('href').split('/').pop();
    $.post("/step2b/api/submit_equivalence_annotation", {
        "claim_id": claim_id,
        "annotations": annos_json
    }, success_callback).fail(enable_submit_button)
}

/**
 * Callback for submission
 */
function success_callback() {
    window.location.href = '/step2b/task_list';
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