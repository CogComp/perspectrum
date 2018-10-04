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