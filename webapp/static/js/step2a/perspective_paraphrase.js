$(document).ready(function () {
    csrfSetup();

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
    let radios = $("input.para");


    // if (checked.length !== $(".equi-cand-container").length) {
    //
    // }
    let unfinished = false;

    radios.each(function() {
        let $this = $(this);
        if ($this.val().length === 0) {
            unfinished = true;
        }

        let el_id = $this.attr('id').split("_");
        let pid = el_id.pop();

        if (pid in annos) {
            annos[pid].push($this.val());
        }
        else {
            annos[pid] = [$this.val()];
        }
    });



    if (unfinished) {
        alert("Please finish all annotations!");
            return enable_submit_button();
    }

    let annos_json = JSON.stringify(annos);
    console.log(annos_json);

    let batch_id = $(location).attr('href').split('/').pop();

    $.post("/step2a/api/submit_annotation", {
        "batch_id": batch_id,
        "annotations": annos_json
    }, success_callback).fail(enable_submit_button)
}

/**
 * Callback for submission
 */
function success_callback() {
    window.location.href = '/step2a/task_list';
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