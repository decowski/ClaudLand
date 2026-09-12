#!/bin/bash
# fetch_theses.sh -- download the publicly available KamLAND / KamLAND-Zen PhD theses.
#
#   ./fetch_theses.sh            # all theses into this directory
#   ./fetch_theses.sh -n         # dry run: list what would be downloaded
#   ./fetch_theses.sh -d dir -s 5
#
# Sources (all checked September 2026, no password needed):
#   * Tohoku University, Research Center for Neutrino Science, the theses that are not behind
#     the collaboration password: https://www.awa.tohoku.ac.jp/Thesis/Doctor_Thesis.html
#   * INSPIRE-HEP full-text copies (https://inspirehep.net, thesis records with KamLAND in the title)
#   * the Stanford Gratta group, the University of Alabama and MIT repositories
# Files are saved as <family>_<given>_<year>.pdf; existing files are skipped, so the script can be
# re-run.  Not available anywhere in the open as far as we could find: the early Alabama theses
# (Mei 2003, McKinny 2003, Djurcic 2004, Classen 2007), Maricic (Hawaii 2005), and the Tohoku theses
# marked "KamLAND collaborator's only" other than Iwamoto and Tajima, which INSPIRE hosts.
set -u
cd "$(dirname "$0")"
DEST="."; SLEEP=2; DRY=0
while getopts "d:s:nh" opt; do
  case $opt in
    d) DEST=$OPTARG ;;
    s) SLEEP=$OPTARG ;;
    n) DRY=1 ;;
    h|*) sed -n 2,17p "$0"; exit 0 ;;
  esac
done
TOHOKU="https://www.awa.tohoku.ac.jp/Thesis/ThesisFile"
INSPIRE="https://inspirehep.net/files"
STANFORD="https://web.stanford.edu/group/grattalab/Theses"

# output file | author | date | institution | title | url
THESES="
iwamoto_toshiyuki_2003.pdf  | Toshiyuki Iwamoto  | 2003-02 | Tohoku       | Measurement of reactor anti-neutrino disappearance in KamLAND | $INSPIRE/a751a9ab96154db78ea2d19f06cf67fc
tajima_osamu_2003.pdf       | Osamu Tajima       | 2003-02 | Tohoku       | Measurement of electron anti-neutrino oscillation parameters with a large volume liquid scintillator detector, KamLAND | $INSPIRE/4b7d9017be54a5d912bcf333d9baafee
ogawa_hiroshi_2003.pdf      | Hiroshi Ogawa      | 2003-10 | Tohoku       | Search for electron anti-neutrinos from the Sun using the KamLAND large volume liquid scintillator detector | $TOHOKU/ogawa_hiroshi_d.pdf
detwiler_jason_2005.pdf     | Jason Detwiler     | 2005-01 | Stanford     | Measurement of neutrino oscillation with KamLAND | $STANFORD/Jason-Detwiler-Thesis.pdf
tolich_nikolai_2005.pdf     | Nikolai Tolich     | 2005-01 | Stanford     | Experimental study of terrestrial electron anti-neutrinos with KamLAND | $STANFORD/Nikolai-Tolich-Thesis.pdf
enomoto_sanshiro_2005.pdf   | Sanshiro Enomoto   | 2005-02 | Tohoku       | Neutrino geophysics and observation of geo-neutrinos at KamLAND | $TOHOKU/enomoto_sanshiro_d.pdf
nakajima_kyo_2005.pdf       | Kyo Nakajima       | 2005-02 | Tohoku       | Measurement of neutrino oscillation parameters with precise calculation of reactor neutrino spectrum at KamLAND | $INSPIRE/56176af486d11012916aac90d0d0043e
batygov_mikhail_2006.pdf    | Mikhail Batygov    | 2006-12 | Tennessee    | Combined study of reactor and terrestrial antineutrinos with KamLAND | $INSPIRE/525d3c53e7e40eddd6672bbb99ae7ac4
dwyer_daniel_2007.pdf       | Daniel Dwyer       | 2007    | UC Berkeley  | Precision measurement of neutrino oscillation parameters with KamLAND | $INSPIRE/207e5a8297a23296ff79b5b9063d7721
tolich_kazumi_2008.pdf      | Kazumi Tolich      | 2008    | Stanford     | Measurement of neutrino oscillation parameters and investigation of uranium and thorium abundances in the Earth using anti-neutrinos | $STANFORD/Kazumi-Tolich-Thesis.pdf
winslow_lindley_2008.pdf    | Lindley Winslow    | 2008    | UC Berkeley  | First solar neutrinos from KamLAND: a measurement of the 8B solar neutrino flux | $INSPIRE/d53c4318fafa03965d08961c69b4436d
keefer_gregory_2009.pdf     | Gregory Keefer     | 2009    | Alabama      | First observation of 7Be solar neutrinos with KamLAND | https://ir.ua.edu/bitstreams/adead02b-7f31-4e61-a71f-a27b06be8350/download
perevozchikov_oleg_2009.pdf | Oleg Perevozchikov | 2009-08 | Tennessee    | Search for electron antineutrinos from the Sun with KamLAND detector | $INSPIRE/215cde810565e6b2b2419c8dc9432d1f
miletic_tatjana_2009.pdf    | Tatjana Miletic    | 2009-06 | Drexel       | Search for the disappearance of a neutron with KamLAND detector | $INSPIRE/44bd8798198e3bd255d77d15bcc6f8e8
zhang_chao_2010.pdf         | Chao Zhang         | 2010    | Caltech      | Precision measurement of neutrino oscillation parameters and investigation of nuclear georeactor hypothesis with KamLAND | $INSPIRE/5f161a34aa09726b5e18a7a6cc85e3c7
odonnell_thomas_2011.pdf    | Thomas O'Donnell   | 2011    | UC Berkeley  | Precision measurement of neutrino oscillation parameters with KamLAND | $INSPIRE/277a0d6a82a06f71cba7480fef7aaccd
grant_christopher_2012.pdf  | Christopher Grant  | 2012    | Alabama      | A Monte Carlo approach to 7Be solar neutrino analysis with KamLAND | https://ir.ua.edu/bitstreams/e37abdec-598a-4618-9e4f-bad8c913eeea/download
gando_azusa_2012.pdf        | Azusa Gando        | 2012-11 | Tohoku       | First results of neutrinoless double beta decay search with KamLAND-Zen | $TOHOKU/gando_azusa_d.pdf
bezerra_thiago_2013.pdf     | Thiago Bezerra     | 2013-07 | Tohoku       | Delta m^2_31 measurement from reactor neutrino oscillation at different baselines | $TOHOKU/BEZERRA_Thiago_d.pdf
takemoto_yasuhiro_2014.pdf  | Yasuhiro Takemoto  | 2014-02 | Tohoku       | Observation of 7Be solar neutrinos with KamLAND | $TOHOKU/takemoto_yasuhiro_d.pdf
yoshida_hisataka_2014.pdf   | Hisataka Yoshida   | 2014-02 | Tohoku       | Limit on Majorana neutrino mass with neutrinoless double beta decay from KamLAND-Zen | $TOHOKU/yoshida_hisataka_d.pdf
sakai_michinari_2016.pdf    | Michinari Sakai    | 2016-12 | Hawaii       | High energy neutrino analysis at KamLAND and application to dark matter search | $INSPIRE/9f369530450d12935a4c49eca80bd51d
matsuda_sayuri_2016.pdf     | Sayuri Matsuda     | 2016-11 | Tohoku       | Search for neutrinoless double-beta decay in 136Xe after intensive background reduction with KamLAND-Zen | $TOHOKU/matsuda_sayuri_d.pdf
li_aobo_2020.pdf            | Aobo Li            | 2020    | Boston U.    | The Tao and Zen of neutrinos: neutrinoless double beta decay in KamLAND-Zen 800 | $INSPIRE/be8d574b0c22b9a5867a0f954e00b4e7
fraker_suzannah_2022.pdf    | Suzannah Fraker    | 2022-05 | MIT          | Deep learning for the KamLAND-Zen search for neutrinoless double beta decay | $INSPIRE/4011275f310b1cb4ac65ea964d4f4834
abe_seisho_2023.pdf         | Seisho Abe         | 2023-02 | Tohoku       | Measurement of the strangeness axial coupling constant using neutral current quasi-elastic interactions of atmospheric neutrinos at KamLAND | $TOHOKU/abe_seisho_d.pdf
smolsky_joseph_2023.pdf     | Joseph Smolsky     | 2023-09 | MIT          | Ion source development for IsoDAR and multi-messenger astrophysics with KamLAND | https://dspace.mit.edu/server/api/core/bitstreams/fa966e44-3923-4963-85d2-49bdbefedcb6/content
eizuka_minori_2026.pdf      | Minori Eizuka      | 2026-02 | Tohoku       | Search for astrophysical antineutrinos with deep neural-network-based event identification in the full KamLAND dataset | $TOHOKU/eizuka_minori_d.pdf
"

trim() { local v=$1; v=${v#"${v%%[![:space:]]*}"}; v=${v%"${v##*[![:space:]]}"}; printf '%s' "$v"; }
mkdir -p "$DEST"
n_ok=0; n_skip=0; n_fail=0
while IFS='|' read -r file author date inst title url; do
  file=$(trim "$file"); [ -z "$file" ] && continue
  author=$(trim "$author"); date=$(trim "$date"); inst=$(trim "$inst"); title=$(trim "$title"); url=$(trim "$url")
  out="$DEST/$file"
  if [ -s "$out" ]; then n_skip=$((n_skip+1)); continue; fi
  printf "%-28s %-18s %-7s %-12s %s\n" "$file" "$author" "$date" "$inst" "$title"
  [ "$DRY" = 1 ] && continue
  if curl -sSL --fail --retry 3 --retry-delay 5 -A "Mozilla/5.0 (fetch_theses.sh)" "$url" -o "$out.part" \
     && head -c 5 "$out.part" | grep -q '%PDF'; then
    mv "$out.part" "$out"; n_ok=$((n_ok+1))
    echo "    -> $(du -h "$out" | cut -f1)"
  else
    rm -f "$out.part"; n_fail=$((n_fail+1)); echo "    FAILED: $url" >&2
  fi
  sleep "$SLEEP"
done <<< "$THESES"

if [ "$DRY" = 1 ]; then echo "(dry run, $n_skip already present)"
else echo "downloaded $n_ok, skipped $n_skip already present, failed $n_fail -> $DEST/"; fi
[ "$n_fail" = 0 ]
