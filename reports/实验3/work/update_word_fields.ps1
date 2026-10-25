param([string]$Path)
$ErrorActionPreference='Stop'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
$word.AutomationSecurity=3
$doc=$null
try {
    $doc=$word.Documents.Open($Path,$false,$false)
    $doc.Repaginate()
    $null=$doc.Fields.Update()
    foreach($toc in $doc.TablesOfContents) {
        $toc.Update()
        $toc.Range.Font.Name='Times New Roman'
        $toc.Range.Font.NameFarEast='宋体'
        $toc.Range.Font.Size=10.5
        $toc.Range.ParagraphFormat.LineSpacingRule=4
        $toc.Range.ParagraphFormat.LineSpacing=15
        $toc.Range.ParagraphFormat.SpaceBefore=0
        $toc.Range.ParagraphFormat.SpaceAfter=0
        $toc.Range.ParagraphFormat.DisableLineHeightGrid=$true
        $toc.Range.ParagraphFormat.KeepWithNext=$false
        $toc.Range.ParagraphFormat.KeepTogether=$true
    }
    $doc.Repaginate()
    foreach($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    foreach($section in $doc.Sections) {
        foreach($footer in $section.Footers) { $null=$footer.Range.Fields.Update() }
    }
    $doc.Save()
    Write-Output ('Pages: '+$doc.ComputeStatistics(2))
    Write-Output ('Tables: '+$doc.Tables.Count)
    Write-Output ('Figures: '+$doc.InlineShapes.Count)
} finally {
    if ($null -ne $doc) { $doc.Close(0) }
    $word.Quit()
}
