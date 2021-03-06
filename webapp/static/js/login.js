$(document).ready(function() {
    $('#btn-login').click(login);
});

/**
 * TODO: django rotates CSRF token after login(), so if you click login twice, CSRF token check will fail.
 * https://gist.github.com/j4mie/9055969
 */
function login() {
    csrfSetup();
    let user_id = $('#user-id').val();
    if (!user_id) {
        alert("Please enter your name!")
    }
    else {
        $.post("/api/auth_login/", {
            "username": user_id,
        }, login_callback);
    }
}

function login_callback() {
    var url = new URL(window.location.href);
    var redir = url.searchParams.get("next");
    if (redir) {
        window.location.href = redir;
    }
    else {
        window.location.href = '/human_eval/2';
    }
}