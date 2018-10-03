$(document).ready(function () {
    csrfSetup();
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
 * Helper function -- check whether a HTTP method is safe from CSRF
 * @param method HTTP method name
 * @returns {boolean} Yes if safe, no otherwise
 */
function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

/**
 * Set ajax to include CSRF token in request header every time
 * Call this first!
 */
function csrfSetup() {
    var csrftoken = $("[name=csrfmiddlewaretoken]").val();
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}

var rel_mapping = {
    'sup': 'S',
    'und': 'U',
    'irr': 'I',
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