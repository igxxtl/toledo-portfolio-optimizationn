$ErrorActionPreference='Stop'
$base=Join-Path $PSScriptRoot '..'
$source='C:\Users\igor.toledo_visagio\Downloads\reavaliacao_102_casos.csv'
$out=Join-Path $PSScriptRoot 'resultado_reavaliacao'
$expected=@(Import-Csv (Join-Path $PSScriptRoot 'divergencias_humano_ia.csv'))
$incoming=@(Import-Csv -LiteralPath $source -Delimiter ';')
if($incoming.Count -ne 102 -or @($incoming.sample_id | Sort-Object -Unique).Count -ne 102){throw 'Esperados 102 IDs únicos.'}
$lookup=@{};foreach($r in $expected){$lookup[$r.sample_id]=$r}
foreach($r in $incoming){
 if(-not $lookup.ContainsKey($r.sample_id)){throw 'ID inesperado'}
 foreach($col in @('ticker','titulo','data','rotulo_humano_original','rotulo_ia','rotulo_modelo_original','justificativa_ia')){if($r.$col -cne $lookup[$r.sample_id].$col){throw "Identidade alterada: $($r.sample_id), $col"}}
 if($r.rotulo_reavaliado -notin @('positivo','negativo','neutro') -or $r.status_revisao -ne 'reavaliado_pelo_usuario'){throw 'Decisão ausente ou inválida'}
}
$review=@{};foreach($r in $incoming){$review[$r.sample_id]=$r}
$original=@{};Import-Csv (Join-Path $base 'gabarito_modelo.csv') | ForEach-Object {$original[$_.sample_id]=$_.sentimento}
$all=@(foreach($h in (Import-Csv (Join-Path $base 'rotulacao_humana.csv'))){$rev=$review[$h.sample_id];[pscustomobject][ordered]@{sample_id=$h.sample_id;ticker=$h.ticker;titulo=$h.titulo;data=$h.data;rotulo_humano_inicial=$h.rotulo_avaliador_1;rotulo_humano_revisado=$(if($rev){$rev.rotulo_reavaliado}else{$h.rotulo_avaliador_1});rotulo_modelo_original=$original[$h.sample_id];foi_reavaliado=[bool]$rev;observacao_revisao=$(if($rev){$rev.observacao_revisao}else{''});reavaliado_em=$(if($rev){$rev.reavaliado_em}else{''})}})
if($all.Count -ne 300){throw 'Amostra final incompleta'}
$classes=@('positivo','negativo','neutro');$conf=@{};$scores=@()
foreach($c in $classes){$conf[$c]=@{};foreach($d in $classes){$conf[$c][$d]=@($all | Where-Object {$_.rotulo_humano_revisado -eq $c -and $_.rotulo_modelo_original -eq $d}).Count};$support=@($all | Where-Object rotulo_humano_revisado -eq $c).Count;$pred=@($all | Where-Object rotulo_modelo_original -eq $c).Count;$tp=$conf[$c][$c];$scores += [pscustomobject]@{classe=$c;suporte=$support;precisao=$tp/[double]$pred;revocacao=$tp/[double]$support;f1=2*$tp/[double]($support+$pred)}}
$matches=@($all | Where-Object {$_.rotulo_humano_revisado -eq $_.rotulo_modelo_original}).Count
$expectedAgreement=0.0;foreach($c in $classes){$expectedAgreement+=@($all | Where-Object rotulo_humano_revisado -eq $c).Count*@($all | Where-Object rotulo_modelo_original -eq $c).Count/90000.0}
$changed=@($all | Where-Object {$_.rotulo_humano_inicial -ne $_.rotulo_humano_revisado}).Count
$report=[ordered]@{n=300;n_reavaliados=102;n_mantidos_sem_reavaliacao=198;rotulos_alterados=$changed;rotulos_mantidos_na_reavaliacao=102-$changed;concordancia_modelo_original=$matches/300.0;acordos=$matches;macro_f1=($scores | Measure-Object f1 -Average).Average;kappa=($matches/300.0-$expectedAgreement)/(1-$expectedAgreement);referencia_majoritaria=($scores | Measure-Object suporte -Maximum).Maximum/300.0;metricas_por_classe=$scores;matriz=$conf;metodo='Revisão pelo mesmo humano de 102 divergências após avaliação por IA; possibilidade de consultar rótulos e justificativas. Não é avaliação humana independente ou necessariamente cega; 198 rótulos iniciais mantidos.';observacoes='Campo de observação do caso 39 contém o texto 1; preservado literalmente.';sha256_arquivo_importado=(Get-FileHash $source).Hash}
New-Item -ItemType Directory -Path $out -Force | Out-Null
Copy-Item -LiteralPath $source -Destination (Join-Path $out 'reavaliacao_102_recebida.csv')
$all | Export-Csv (Join-Path $out 'rotulacao_300_revisada.csv') -NoTypeInformation -Encoding utf8BOM
$scores | Export-Csv (Join-Path $out 'metricas_por_classe.csv') -NoTypeInformation -Encoding utf8BOM
$matrixRows=foreach($c in $classes){[pscustomobject]@{humano_revisado=$c;modelo_positivo=$conf[$c].positivo;modelo_negativo=$conf[$c].negativo;modelo_neutro=$conf[$c].neutro}}
$matrixRows | Export-Csv (Join-Path $out 'matriz_confusao.csv') -NoTypeInformation -Encoding utf8BOM
$report | ConvertTo-Json -Depth 7 | Set-Content (Join-Path $out 'avaliacao_revisada.json') -Encoding utf8
$report | ConvertTo-Json -Depth 7
