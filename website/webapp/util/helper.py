def get_all_claim_title_id(json_data):
    """
    Helper function for claims.html.
    Get claim/id pairs from raw json data
    id is used for indexing
    :return: a list of (claim, id) tuples
    """

    # TODO: will add unique id to each claim in raw json
    id = 0
    result = []
    for claim in json_data:
        result.append((claim["claim_title"], str(id)))
        id += 1

    return result


def get_claim_given_id(json_data, claim_id):
    """
    Helper function.
    Given id of claim, return the corresponding claim from raw json data

    :param json_data:
    :param claim_id: unique identifier of a claim
    :return: a claim object
    """
    claim_id = int(claim_id)
    return json_data[claim_id]
