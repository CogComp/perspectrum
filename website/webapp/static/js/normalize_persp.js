$(document).ready(function () {
    csrfSetup();

    $("#hideshow").click(function() {
        // assumes element with id='button'
        $(".rel-demo").toggle();
    });

    $('#rel_option_clear').click(function() {
        $('.rel_option').prop('checked', false);
    });
    $('#rel_option_submit').click(submit);
});

var rel_mapping = {
    'sup': 'S',
    'und': 'U',
    'irr': 'I',
    'ns': 'N',
};

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
            let rel = rel_mapping[el_id.pop()];
            annos.push(persp_id + ',' + rel);
        });
    });

    if (el_persps.length !== annos.length) {
        alert("Please annotate all questions!");
        return;
    }
    else {
        let claim_id = $(location).attr('href').split('/').pop();
        $.post("/api/submit_rel_anno/", {
            "claim_id": claim_id,
            "annotations": annos
        }, submit_callback);
    }
}

/**
 * Callback for submission
 * TODO: Change according to AMT protocol, maybe load next page in HIT?
 */
function submit_callback() {
    alert("Submission success!");
}