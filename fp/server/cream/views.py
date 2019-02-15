from copy import copy
from typing import Union

import pandas as pd
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse, HttpRequest
from django.views.decorators.http import require_GET, require_POST
from rdkit.Chem import MolFromMolBlock, MolFromSmiles

from ctrainlib import fplib, rdkit_support
from cream.trained_models import model_manager


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """
    Default view of the server. Can be called in a browser to see
    an overview over all available models. Only safe requests are
    allowed.

    Parameters
    ----------
    request : HttpRequest
        Incoming GET request as HttpRequest instance

    Returns
    -------
    HttpResponse
        HttpResponse instance containing all loaded and available models
    """
    return HttpResponse('Available models:<br><br>' + '<br>'.join(model_manager.models.keys()))


@require_GET
def get_models(request: HttpRequest) -> JsonResponse:
    """
    Returns a list of all available models in JSON format.

    Parameters
    ----------
    request : HttpRequest
        Incoming GET request as HttpRequest instance

    Returns
    -------
    JsonResponse
        JsonResponse instance containing a list of all available models
    """
    return JsonResponse(list(model_manager.models.keys()), safe=False)


@require_POST
def predict(request: HttpRequest) -> Union[HttpResponseBadRequest, JsonResponse]:
    """
    Returns a list of all available models in JSON format.

    Required POST parameters are either 'molblocks' containing a list of molblocks,
    or 'smiles' containing a list auf SMILES. Also required is 'models' with a list
    of model names which should be used for predictions.

    Parameters
    ----------
    request : HttpRequest
        Incoming POST request as HttpRequest instance

    Returns
    -------
    Union[HttpResponseBadRequest, JsonResponse]
        Returns a HttpResponseBadRequest instance if any required parameters
        are missing or if an error occurs. Otherwise a JsonResponse instance
        containing all predictions and probabilities (if available) is returned.
    """

    if 'molblocks' in request.POST and 'smiles' in request.POST \
            or 'molblocks' not in request.POST and 'smiles' not in request.POST:
        return HttpResponseBadRequest('molblocks OR smiles have to be provided')
    if 'models' not in request.POST:
        return HttpResponseBadRequest('model(s) have to be provided')
    model_names = request.POST.getlist('models')
    model_names = set(model_names)

    descs = []
    fps = []
    for model in model_names:
        if model in model_manager.models:
            descs.extend(model_manager.models[model].descriptors)
            fps.extend(model_manager.models[model].fingerprints)
        else:
            return HttpResponseBadRequest(f'{model} not available')

    descs = list(set(descs))
    fps = list(set(fps))
    mols = []
    if 'molblocks' in request.POST:
        molblocks = request.POST.getlist('molblocks')
        if not molblocks or molblocks == ['']:
            return HttpResponseBadRequest('List of molblocks is empty')
        for mb in molblocks:
            if mb == '':
                mols.append(None)
                break
            mols.append(MolFromMolBlock(mb))
    else:
        smiles = request.POST.getlist('smiles')
        if not smiles or smiles == ['']:
            return HttpResponseBadRequest('List of SMILES is empty')
        for smi in smiles:
            if smi == '':
                mols.append(None)
                break
            mols.append(MolFromSmiles(smi))

    if None in mols:
        return HttpResponseBadRequest('molblocks or smiles contain invalid entries')

    df = pd.DataFrame.from_dict(dict(ROMol=mols))
    df = rdkit_support.compute_descriptors(df, descs)
    bad_ix, bad_desc = rdkit_support.filter_descriptor_values(df[descs])
    if len(bad_ix) > 0:
        df.drop(index=bad_ix, inplace=True)
        if len(df) == 0:
            return HttpResponseBadRequest('Every molecule leads to bad descriptor values')
    df = fplib.compute_fingerprints(df, fps)
    df.drop(columns='ROMol', inplace=True)

    res = {ix: {} for ix in df.index}
    for model in model_names:
        m = model_manager.models[model]
        cols = copy(m.descriptors)
        for fp in m.fingerprints:
            for fp_col in df.columns[len(descs):]:
                if fp_col.startswith(f'{fp.alias}['):
                    cols.append(fp_col)
        pred = m.predict(df[cols])
        pred.columns = [col.replace(f'{model}_', '') for col in pred.columns]
        pred = pred.to_dict('index')
        for ix in pred:
            res[ix][model] = pred[ix]

    return JsonResponse(res)
