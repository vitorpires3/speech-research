#!/usr/bin/perl
# ./seg2lbl.v2b.pl fichier.seg [output_folder]
# creates: output_folder/fichier.lblmore, output_folder/fichier.wrd, output_folder/fichier.TextGrid

use strict;
use warnings;
#use open qw(:std :encoding(UTF-8));
use File::Basename;

my $segt = $ARGV[0] or die "usage: $0 fichier.seg [output_folder]\n";
my $output_dir = $ARGV[1] || ".";

my $file = basename($segt, ".seg");
my $input_dir = dirname($segt);

unless (-d $output_dir || mkdir($output_dir, 0755)) {
    die "cannot create output directory $output_dir: $!";
}

my $lblmore = "$output_dir/$file.lblmore";
my $wrd     = "$output_dir/$file.wrd";
my $tg      = "$output_dir/$file.TextGrid";

print "input=$segt -- output_dir=$output_dir -- lblmore=$lblmore -- wrd=$wrd -- tg=$tg\n";


#open(my $LBLM, "|sort -n >$lblmore") or die "cannot open $lblmore: $!";
#open(my $WRD,  ">$wrd")              or die "cannot open $wrd: $!";
#open(my $SEGT, "<$segt")             or die "cannot open $segt: $!";

#---------------------------
open(my $LBLM, "|sort -n >$lblmore")
    or die "cannot open $lblmore: $!";

binmode($LBLM, ":encoding(ISO-8859-1)");

open(
    my $WRD,
    ">:encoding(ISO-8859-1)",
    $wrd
) or die "cannot open $wrd: $!";

open(
    my $SEGT,
    "<:encoding(ISO-8859-1)",
    $segt
) or die "cannot open $segt: $!";
#--------------------------

my $timedeb = 0;
my $timefin = 0;

my $wdur      = 0;
my $word      = "#START";
my $nbph      = 0;
my $meandur   = 0;
my $wordphon  = "#START";
my @ph        = ();
my @dur       = ();

my $word_start;
my $prev_phone_end;
my $last_phone_end;

my @phones;  # [start, end, phone]
my @words;   # [start, end, word]

printf $WRD "word phonetisation duration nbPhones durMoyPhon\n";

while (<$SEGT>) {
    if (/^\#@ sid=.*\-([0-9\.]+)\-([0-9\.]+)\s*$/) {
        $timedeb = $1;
        $timefin = $2;
        next;
    }

    if (/^\#@ word=(\S+) /) {
        if ($nbph) {
            $meandur = $wdur / $nbph;
            push @words, [ $word_start, $prev_phone_end, $word ]
                if defined $word_start;

            printf $WRD "%s %s %5.0f %d (%4.0f)",
                $word, $wordphon, $wdur*1000, $nbph, $meandur*1000;
            for my $i (0 .. $#ph) {
                my $d_ms = $dur[$i]*1000;
                printf $WRD " %s=%5.0f", $ph[$i], $d_ms;
            }
            printf $WRD "\n";
        }

        $word     = $1;
        $wdur     = 0;
        $nbph     = 0;
        $wordphon = "";
        @ph       = ();
        @dur      = ();
        undef $word_start;
        next;
    }

    if (/^[^\#].*\:(\S)(\S)(\S)\S*\s+[0-9]+\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+/) {
        my $ctxGauche = $1;
        my $phone     = $2;
        my $ctxDroit  = $3;
        my $f_start   = $4;
        my $p1        = $5;
        my $p2        = $6;
        my $p3        = $7;

        my $time1 = ($f_start - 1) / 100.0;
        my $duree = ($p1 + $p2 + $p3) / 100.0;

        $wdur   += $duree;
        $nbph   += 1;
        push @ph,  $phone;
        push @dur, $duree;

        $wordphon .= $phone;

        my $abs_start = $timedeb + $time1;
        my $abs_end   = $abs_start + $duree;
        $last_phone_end = $abs_end;

        push @phones, [ $abs_start, $abs_end, $phone ];

        if (!defined $word_start) {
            $word_start = $abs_start;
        }
        $prev_phone_end = $abs_end;

        printf $LBLM "%0.02f %s\n", $abs_start, $phone;
        next;
    }
}

close $SEGT;
close $LBLM;

if ($nbph) {
    $meandur = $wdur / $nbph;
    push @words, [ $word_start, $prev_phone_end, $word ]
        if defined $word_start;

    printf $WRD "%s %s %5.0f %d (%4.0f)",
        $word, $wordphon, $wdur*1000, $nbph, $meandur*1000;
    for my $i (0 .. $#ph) {
        my $d_ms = $dur[$i]*1000;
        printf $WRD " %s=%5.0f", $ph[$i], $d_ms;
    }
    printf $WRD "\n";
}

close $WRD;

@phones = sort { $a->[0] <=> $b->[0] } @phones;
@words  = sort { $a->[0] <=> $b->[0] } @words;

my $xmin = @phones ? $phones[0]->[0] : $timedeb;
my $xmax = @phones ? $last_phone_end : $timefin;
$xmax = $timefin if $timefin > $xmax;

#open(my $TG, ">$tg") or die "cannot open $tg: $!";
#------------------
open(
    my $TG,
    ">:encoding(ISO-8859-1)",
    $tg
) or die "cannot open $tg: $!";
#------------------

print $TG "File type = \"ooTextFile\"\n";
print $TG "Object class = \"TextGrid\"\n\n";
print $TG "xmin = $xmin\n";
print $TG "xmax = $xmax\n";
print $TG "tiers? <exists>\n";
print $TG "size = 2\n";
print $TG "item []:\n";

print $TG "    item [1]:\n";
print $TG "        class = \"IntervalTier\"\n";
print $TG "        name = \"phones\"\n";
print $TG "        xmin = $xmin\n";
print $TG "        xmax = $xmax\n";
print $TG "        intervals: size = ".scalar(@phones)."\n";
my $i = 1;
for my $p (@phones) {
    my ($s, $e, $ph) = @$p;
    print $TG "        intervals [$i]:\n";
    print $TG "            xmin = $s\n";
    print $TG "            xmax = $e\n";
    print $TG "            text = \"$ph\"\n";
    $i++;
}

print $TG "    item [2]:\n";
print $TG "        class = \"IntervalTier\"\n";
print $TG "        name = \"words\"\n";
print $TG "        xmin = $xmin\n";
print $TG "        xmax = $xmax\n";
print $TG "        intervals: size = ".scalar(@words)."\n";
$i = 1;
for my $w (@words) {
    my ($ws, $we, $lab) = @$w;
    print $TG "        intervals [$i]:\n";
    print $TG "            xmin = $ws\n";
    print $TG "            xmax = $we\n";
    print $TG "            text = \"$lab\"\n";
    $i++;
}

close $TG;

exit 0;