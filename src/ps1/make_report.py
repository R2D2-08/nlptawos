"""Generate the actual-results PS-1 report in PDF and/or Markdown.

All quantitative results are loaded from the experiment's JSON/CSV files.
Do not generate a measured-results report before running the experiment.
"""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether


# Identifying information copied from the provided PS1_Final_Report.pdf.
DEFAULT_GROUP = 'Group 2'
DEFAULT_MEMBERS = 'Aditya, Athul, Johan, Kiran, Omar'
DEFAULT_GITHUB = 'https://github.com/R2D2-08/nlptawos'

# Both report formats use the same plot files from the measured run.
PLOTS = [
    ('Five-fold validation MAE', 'cross_validation_mae.png'),
    ('Five-fold validation MdAE', 'cross_validation_mdae.png'),
    ('Actual versus predicted story points', 'test_actual_vs_predicted.png'),
    ('Held-out absolute errors', 'test_absolute_errors.png'),
    ('Final CNN training loss', 'training_loss.png'),
]

LIMITATIONS = [
    'Within-project estimates may not transfer to other projects with different jargon and story point scales.',
    'Random splitting mixes chronological periods; future deployment may behave differently.',
    'Story points are subjective and noisy, and some post-estimation edit flags may be missing or unknown.',
    'Frozen BERT is not software-domain fine-tuned, and token truncation may omit important issue details.',
    'Five-fold validation and one test split estimate performance imperfectly; more seeds/projects could increase confidence.',
    'Repeated wording and correlated issues may remain even after exact duplicate filtering.',
]

REFERENCES = [
    'Tawosi et al. (2022), A Versatile Dataset of Agile Open Source Software Projects, DOI 10.1145/3524842.3528029.',
    'TAWOS dataset: https://doi.org/10.5522/04/21308124',
    'Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv:1810.04805.',
]



def p(text, style):
    return Paragraph(escape(str(text)), style)


def table(rows, widths, doc_width):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#e6edf4')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#152f4b')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f5f8fa')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9db1c3')),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),8),
        ('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),
    ]))
    return t


def build(run_root, output, group=DEFAULT_GROUP, members=DEFAULT_MEMBERS, github=DEFAULT_GITHUB):
    base = Path(run_root)
    r, data = base/'results', base/'data'
    needed = [r/'summary.json',r/'cv_summary.csv',r/'heldout_test_results.csv',
              r/'five_fold_results.csv',data/'audit.json',r/'reproduction.json']
    missing = [str(x) for x in needed if not x.is_file()]
    if missing:
        raise FileNotFoundError('Run the ACTUAL experiment first; missing: ' + ', '.join(missing))
    summary = json.loads((r/'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((data/'audit.json').read_text(encoding='utf-8'))
    reprod = json.loads((r/'reproduction.json').read_text(encoding='utf-8'))
    conf = reprod['config']
    val = pd.read_csv(r/'cv_summary.csv')
    fold = pd.read_csv(r/'five_fold_results.csv')
    test = pd.read_csv(r/'heldout_test_results.csv')
    project = summary['project']
    chosen = summary['chosen_cnn']
    cnn = test.loc[test['model'] == chosen].iloc[0]
    baseline = test.loc[test['model'] == 'dummy_median'].iloc[0]
    ridge = test.loc[test['model'] == 'ridge_bert_mean'].iloc[0]
    state = 'lower' if cnn['MAE'] < baseline['MAE'] else ('higher' if cnn['MAE'] > baseline['MAE'] else 'equal')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Cover', parent=styles['Title'], fontSize=19, leading=25, spaceAfter=20,
                              textColor=colors.HexColor('#153c62'), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='PS1Body', parent=styles['BodyText'], fontSize=9.6, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='PS1Small', parent=styles['BodyText'], fontSize=8, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name='PS1Heading', parent=styles['Heading2'], spaceBefore=12, spaceAfter=8,
                              textColor=colors.HexColor('#153c62')))
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=45, rightMargin=45,
                            topMargin=42, bottomMargin=44, title='PS1: BERT Within-Project Story Points')
    W = doc.width
    body = []
    def addpara(text, name='PS1Body'):
        body.append(p(text, styles[name]))
    def heading(text):
        body.append(p(text, styles['PS1Heading']))
    body.append(p('BERT-Based Word Embeddings for Within-Project Software Story Point Estimation', styles['Cover']))
    addpara('Project Statement 1 (PS-1) | Natural Language Processing')
    addpara('Group: ' + (group or '[ADD GROUP NUMBER]'))
    addpara('Members: ' + (members or '[ADD GROUP MEMBERS]'))
    addpara('GitHub repository: ' + (github or '[ADD PUBLIC REPOSITORY URL]'))
    addpara('Project data: ' + str(project) + ' (TAWOS v1.1)')
    addpara('Results below come from supplied run files. Verify real TAWOS provenance before publication.', 'PS1Small')
    heading('Abstract')
    addpara(f'We evaluated contextual BERT token representations for predicting software story points using a random within-project split on the {project} project. '
            f'The experiment reserved {summary["heldout_test_n"]} of {summary["n"]} eligible issues for final testing and ran five-fold cross-validation on the development set. '
            f'A frozen pretrained BERT encoder produced token embeddings for a three-layer convolutional regression head. '
            f'The final chosen CNN obtained held-out MAE {cnn["MAE"]:.4f} and MdAE {cnn["MdAE"]:.4f}. '
            f'The constant-median baseline yielded MAE {baseline["MAE"]:.4f} and MdAE {baseline["MdAE"]:.4f}. '
            'Detailed validation and test results are presented separately.')
    heading('1. Research question')
    addpara('How effectively do BERT-based contextual token embeddings predict story point values from the title and description of an issue, when training and testing use distinct issues from the same project under a random split? This is supervised numerical regression.')
    heading('2. Data collection and preprocessing')
    addpara('Dataset: Tawosi et al., TAWOS v1.1 (DOI 10.5522/04/21308124). The official database is MySQL; the extractor joins Issue and Project, using Title and Description_Text as input and Story_Point as the target. Issues explicitly changed after estimation, empty texts, invalid labels, and repeated exact descriptions are filtered. Unknown change flags remain a potential limitation.')
    audits = [('Audit stage','Count'),
              ('Source issues with project key',str(audit['raw_project_issues'])),
              ('Positive numeric labels',str(audit['with_positive_numeric_story_points'])),
              ('After explicit edit filtering',str(audit['after_excluding_explicit_post_estimate_edits'])),
              ('Final distinct cleaned issues',str(audit['final_unique_issues'])),
              ('Development data (80%)',str(summary['development_n'])),
              ('Final held-out test (20%)',str(summary['heldout_test_n']))]
    body.append(table(audits, [W*.7,W*.3],W))
    heading('3. Representation and architecture')
    addpara(f'Pretrained {conf["model_name"]}, frozen and used for contextual token features. Text is tokenized with padding/truncation at {conf["max_length"]} tokens. '
            f'The token features feed Conv1d layers with {conf["conv_channels"]} output channels, kernel size {conf["kernel_size"]}, ReLU and masked mean pooling; a dense head predicts story points. '
            f'Fixed-epoch training uses SmoothL1 loss, AdamW learning rate {conf["learning_rate"]}, weight decay {conf["weight_decay"]}, '
            f'batch size {conf["train_batch_size"]}, epochs {conf["epochs"]}, seed {conf["seed"]}.')
    heading('4. Validation protocol and selection')
    addpara('The final test set is isolated before model development. Five folds of the 80% development set each take a turn as validation data. CNN dropout candidates are chosen by lowest mean validation MAE, with mean MdAE as tie-breaker. The chosen CNN is refitted on all development examples and evaluated once on the untouched held-out test. Baselines are constant training median and Ridge regression on average BERT token features.')
    body.append(PageBreak())
    heading('5. Five-fold validation results (NOT final test)')
    rows = [['Model','MAE mean +/- SD','MdAE mean +/- SD']]
    for _,v in val.iterrows():
        rows.append([p(v['model'],styles['PS1Small']),
                     f'{v["MAE_mean"]:.4f} +/- {v["MAE_std"]:.4f}',
                     f'{v["MdAE_mean"]:.4f} +/- {v["MdAE_std"]:.4f}'])
    body.append(table(rows,[W*.44,W*.28,W*.28],W))
    addpara(f'The selected CNN setting based on development cross-validation was {chosen}. Individual fold observations are saved in five_fold_results.csv.')
    heading('6. Final held-out test results (evaluated separately)')
    rows = [['Model','MAE','MdAE']]
    for _,row in test.iterrows():
        rows.append([p(row['model'],styles['PS1Small']), f'{row["MAE"]:.4f}',f'{row["MdAE"]:.4f}'])
    body.append(table(rows,[W*.5,W*.25,W*.25],W))
    heading('7. Findings')
    addpara(f'The selected CNN MAE ({cnn["MAE"]:.4f}) is {state} than the constant-median baseline MAE ({baseline["MAE"]:.4f}). '
            f'The BERT-mean Ridge comparison has held-out MAE {ridge["MAE"]:.4f}. '
            'These are observed associations in one project/split, not proof that BERT causes better estimates across settings. Compare MdAE and fold variability, and inspect large errors before making strong conclusions.')
    for title, name in PLOTS:
        file = r/'plots'/name
        if file.exists():
            body.append(PageBreak())
            heading(title)
            im = Image(str(file))
            scale = min(W/im.imageWidth,(A4[1]-160)/im.imageHeight)
            im.drawWidth, im.drawHeight = im.imageWidth*scale, im.imageHeight*scale
            body.append(im)
    body.append(PageBreak())
    heading('8. Limitations and threats to validity')
    for point in LIMITATIONS:
        addpara('- ' + point)
    heading('9. Conclusion')
    addpara(f'In this observed within-project experiment, {chosen} obtained a held-out MAE of {cnn["MAE"]:.4f} and MdAE of {cnn["MdAE"]:.4f} on {summary["heldout_test_n"]} test examples. '
            f'These values should be interpreted relative to label distribution, fold variation, and baseline comparisons. '
            'Potential next steps include longer input lengths, additional independent project replications and a temporal-split study.')
    heading('References')
    for ref in REFERENCES + ['Implementation and full replication commands: ' + (github or '[ADD GitHub repository URL]')]:
        addpara(ref,'PS1Small')
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#667789'))
        canvas.drawRightString(A4[0]-46,27,f'PS-1 | {project} | Page {doc.page}')
    doc.build(body,onFirstPage=footer,onLaterPages=footer)
    print(f'Report written from measured data: {output}')


def _md_cell(value):
    """Protect Markdown table cells from accidental newlines/pipes."""
    return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ')


def _md_table(rows):
    width = len(rows[0])
    lines = ['| ' + ' | '.join(_md_cell(x) for x in rows[0]) + ' |',
             '| ' + ' | '.join(['---'] * width) + ' |']
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(_md_cell(x) for x in row) + ' |')
    return '\n'.join(lines)


def build_markdown(run_root, output, group=DEFAULT_GROUP, members=DEFAULT_MEMBERS,
                   github=DEFAULT_GITHUB):
    """Write a Markdown counterpart sourced from the exact files used for the PDF.

    Available plots are copied into a sibling <report>_assets/ folder so
    image references remain relative and portable with the document.
    """
    base = Path(run_root)
    r, data = base / 'results', base / 'data'
    needed = [r/'summary.json', r/'cv_summary.csv', r/'heldout_test_results.csv',
              r/'five_fold_results.csv', data/'audit.json', r/'reproduction.json']
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError('Run the ACTUAL experiment first; missing: ' + ', '.join(missing))

    summary = json.loads((r/'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((data/'audit.json').read_text(encoding='utf-8'))
    reprod = json.loads((r/'reproduction.json').read_text(encoding='utf-8'))
    conf = reprod['config']
    val = pd.read_csv(r/'cv_summary.csv')
    # Deliberately required above: provides the actual fold observations.
    test = pd.read_csv(r/'heldout_test_results.csv')
    project, chosen = summary['project'], summary['chosen_cnn']
    cnn = test.loc[test['model'] == chosen].iloc[0]
    baseline = test.loc[test['model'] == 'dummy_median'].iloc[0]
    ridge = test.loc[test['model'] == 'ridge_bert_mean'].iloc[0]
    state = 'lower' if cnn['MAE'] < baseline['MAE'] else ('higher' if cnn['MAE'] > baseline['MAE'] else 'equal')

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        '# BERT-Based Word Embeddings for Within-Project Software Story Point Estimation',
        '**Project Statement 1 (PS-1) | Natural Language Processing**',
        f'**Group:** {group or "[ADD GROUP NUMBER]"}  \n'
        f'**Members:** {members or "[ADD GROUP MEMBERS]"}  \n'
        f'**GitHub repository:** {github or "[ADD PUBLIC REPOSITORY URL]"}  \n'
        f'**Project data:** {project} (TAWOS v1.1)',
        '> Results below come from supplied run files. Verify real TAWOS provenance before publication.',
        '## Abstract',
        f'We evaluated contextual BERT token representations for predicting software story points using a random '
        f'within-project split on the {project} project. The experiment reserved {summary["heldout_test_n"]} of '
        f'{summary["n"]} eligible issues for final testing and ran five-fold cross-validation on the development set. '
        f'A frozen pretrained BERT encoder produced token embeddings for a three-layer convolutional regression head. '
        f'The final chosen CNN obtained held-out MAE {cnn["MAE"]:.4f} and MdAE {cnn["MdAE"]:.4f}. '
        f'The constant-median baseline yielded MAE {baseline["MAE"]:.4f} and MdAE {baseline["MdAE"]:.4f}. '
        'Detailed validation and test results are presented separately.',
        '## 1. Research question',
        'How effectively do BERT-based contextual token embeddings predict story point values from the title '
        'and description of an issue, when training and testing use distinct issues from the same project '
        'under a random split? This is supervised numerical regression.',
        '## 2. Data collection and preprocessing',
        'Dataset: Tawosi et al., TAWOS v1.1 (DOI 10.5522/04/21308124). The official database is MySQL; '
        'the extractor joins Issue and Project, using Title and Description_Text as input and Story_Point as the target. '
        'Issues explicitly changed after estimation, empty texts, invalid labels, and repeated exact descriptions '
        'are filtered. Unknown change flags remain a potential limitation.',
        _md_table([
            ['Audit stage', 'Count'],
            ['Source issues with project key', audit['raw_project_issues']],
            ['Positive numeric labels', audit['with_positive_numeric_story_points']],
            ['After explicit edit filtering', audit['after_excluding_explicit_post_estimate_edits']],
            ['Final distinct cleaned issues', audit['final_unique_issues']],
            ['Development data (80%)', summary['development_n']],
            ['Final held-out test (20%)', summary['heldout_test_n']],
        ]),
        '## 3. Representation and architecture',
        f'Pretrained {conf["model_name"]}, frozen and used for contextual token features. Text is tokenized '
        f'with padding/truncation at {conf["max_length"]} tokens. The token features feed Conv1d layers '
        f'with {conf["conv_channels"]} output channels, kernel size {conf["kernel_size"]}, ReLU and masked '
        f'mean pooling; a dense head predicts story points. Fixed-epoch training uses SmoothL1 loss, '
        f'AdamW learning rate {conf["learning_rate"]}, weight decay {conf["weight_decay"]}, '
        f'batch size {conf["train_batch_size"]}, epochs {conf["epochs"]}, seed {conf["seed"]}.',
        '## 4. Validation protocol and selection',
        'The final test set is isolated before model development. Five folds of the 80% development set each '
        'take a turn as validation data. CNN dropout candidates are chosen by lowest mean validation MAE, '
        'with mean MdAE as tie-breaker. The chosen CNN is refitted on all development examples and evaluated '
        'once on the untouched held-out test. Baselines are constant training median and Ridge regression '
        'on average BERT token features.',
        '## 5. Five-fold validation results (NOT final test)',
        _md_table([['Model', 'MAE mean +/- SD', 'MdAE mean +/- SD']] + [
            [v['model'], f'{v["MAE_mean"]:.4f} +/- {v["MAE_std"]:.4f}',
             f'{v["MdAE_mean"]:.4f} +/- {v["MdAE_std"]:.4f}'] for _, v in val.iterrows()
        ]),
        f'The selected CNN setting based on development cross-validation was `{chosen}`. '
        'Individual fold observations are saved in `five_fold_results.csv`.',
        '## 6. Final held-out test results (evaluated separately)',
        _md_table([['Model', 'MAE', 'MdAE']] + [
            [row['model'], f'{row["MAE"]:.4f}', f'{row["MdAE"]:.4f}'] for _, row in test.iterrows()
        ]),
        '## 7. Findings',
        f'The selected CNN MAE ({cnn["MAE"]:.4f}) is {state} than the constant-median baseline MAE '
        f'({baseline["MAE"]:.4f}). The BERT-mean Ridge comparison has held-out MAE {ridge["MAE"]:.4f}. '
        'These are observed associations in one project/split, not proof that BERT causes better estimates '
        'across settings. Compare MdAE and fold variability, and inspect large errors before making strong conclusions.',
    ]

    assets = out.parent / (out.stem + '_assets')
    for title, filename in PLOTS:
        source = r / 'plots' / filename
        if source.is_file():
            assets.mkdir(parents=True, exist_ok=True)
            target = assets / filename
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            sections.append(f'### {title}\n\n![{title}]({assets.name}/{filename})')

    sections.extend([
        '## 8. Limitations and threats to validity',
        '\n'.join('- ' + point for point in LIMITATIONS),
        '## 9. Conclusion',
        f'In this observed within-project experiment, {chosen} obtained a held-out MAE of {cnn["MAE"]:.4f} '
        f'and MdAE of {cnn["MdAE"]:.4f} on {summary["heldout_test_n"]} test examples. These values should '
        'be interpreted relative to label distribution, fold variation, and baseline comparisons. '
        'Potential next steps include longer input lengths, additional independent project replications '
        'and a temporal-split study.',
        '## References',
        '\n'.join('- ' + ref for ref in REFERENCES + [
            'Implementation and full replication commands: ' + (github or '[ADD GitHub repository URL]')
        ]),
    ])
    out.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'Markdown report written from measured data: {out}')


def main():
    ap = argparse.ArgumentParser(description='Write reproducible PS1 final PDF and/or Markdown from actual run outputs')
    ap.add_argument('--run', required=True, help='runs/PROJECT folder')
    ap.add_argument('--output', default='submission_results/PS1_Final_Report.pdf', help='PDF output path')
    ap.add_argument('--markdown-output', default=None,
                    help='Markdown output path (default: PDF path with .md extension)')
    ap.add_argument('--format', choices=('pdf', 'markdown', 'both'), default='both',
                    help='Output format; default generates both PDF and Markdown')
    ap.add_argument('--group', default=DEFAULT_GROUP, help='Group identifier')
    ap.add_argument('--members', default=DEFAULT_MEMBERS, help='Comma-separated members')
    ap.add_argument('--github-url', default=DEFAULT_GITHUB, help='Repository URL')
    args = ap.parse_args()

    if args.format in ('pdf', 'both'):
        build(args.run, args.output, args.group, args.members, args.github_url)
    if args.format in ('markdown', 'both'):
        markdown_path = args.markdown_output or str(Path(args.output).with_suffix('.md'))
        build_markdown(args.run, markdown_path, args.group, args.members, args.github_url)


if __name__ == '__main__':
    main()
