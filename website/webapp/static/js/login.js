$(document).ready(function() {
    csrfSetup();  // see csrf_util.js

    $('#btn-login').click(login);
});

function login() {
    user_id = $('#user-id').val();
    pwd = $('#password').val();
    $.post("/api/auth_login/", {
            "user_id": claim_id,
            "annotations": annos
        }, submit_callback);
}