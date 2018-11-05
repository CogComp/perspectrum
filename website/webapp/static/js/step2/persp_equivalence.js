$(document).ready(function () {
    csrfSetup();
    let submitted = false;

    $('#rel_option_clear').click(function() {
        $('.custom-checkbox').prop('checked', false);
    });
    $('#rel_option_submit').click(function (){
        if (!submitted) {
            submit();
        }
        submitted = true
    });
});

/**
 * Submit the annotations to backend
 */
function submit() {

    // Get all annotations
    let el_persps = $('.checkbox');
    let annos = {};
    el_persps.each(function() {
        $(this).find('input:checked').each(function() {
            let el_id = $(this).attr('id').split("_");
            let cand_id = el_id.pop();
            let persp_id = el_id.pop();
            if (persp_id in annos) {
                annos[persp_id].push(cand_id);
            }
            else {
                annos[persp_id] = [cand_id]
            }
        });
    });

    let annos_json = JSON.stringify(annos);

    console.log(annos_json);

    let claim_id = $(location).attr('href').split('/').pop();
    $.post("/step2/api/submit_equivalence_annotation", {
        "claim_id": claim_id,
        "annotations": annos_json
    }, submit_callback);
}

/**
 * Callback for submission
 */
function submit_callback() {
    window.location.href = '/step2/task_list';
}